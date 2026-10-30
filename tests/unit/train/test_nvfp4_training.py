import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from prime_rl.configs.trainer import NVFP4Config
from prime_rl.trainer.models.layers.nvfp4_linear import (
    NVFP4Linear,
    initialize_nvfp4_runtime,
    replace_linear_with_nvfp4_linear,
)
from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bucket_updates_and_cpu_resume_match_adamw(dtype):
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.Linear(128, 128)).cuda().to(dtype)
    reference = copy.deepcopy(model)
    plain = torch.optim.AdamW(reference.parameters(), lr=1e-3, foreach=False)
    wrapped = CPUOffloadOptimizer(torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False), bucket_bytes=140000)
    for step in range(5):
        for a, b in zip(reference.parameters(), model.parameters()):
            grad = torch.randn_like(a)
            a.grad = grad.clone()
            b.grad = grad.clone()
        plain.step()
        wrapped.step()
        assert all(torch.equal(a, b) for a, b in zip(reference.parameters(), model.parameters()))
        assert all(v.device.type == "cpu" for s in wrapped.state.values() for v in s.values())
        if step == 2:
            wrapped.load_state_dict(copy.deepcopy(wrapped.state_dict()))


def test_bucket_checkpoint_template_stays_on_cpu():
    model = torch.nn.Linear(128, 128, device="cuda")
    optim = CPUOffloadOptimizer(torch.optim.AdamW(model.parameters()), bucket_bytes=140000)
    optim.prepare_checkpoint_save()
    assert all(v.device.type == "cpu" for s in optim.state.values() for v in s.values())
    assert all(s["step"].item() == 0 for s in optim.state.values())


def test_nvfp4_meta_parameters_and_selection():
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.gate_proj = torch.nn.Linear(128, 256, bias=False, device="meta", dtype=torch.bfloat16)
    original = model.mlp.gate_proj.weight
    keys = list(model.state_dict())
    replace_linear_with_nvfp4_linear(model, NVFP4Config())
    assert model.mlp.gate_proj.weight is original
    assert list(model.state_dict()) == keys
    model.to_empty(device="cuda")
    initialize_nvfp4_runtime(model)
    assert model.mlp.gate_proj._sr_seed.item() == model.mlp.gate_proj.sr_seed


@pytest.mark.parametrize("compiled", [False, True])
def test_nvfp4_forward_backward_padding_and_checkpoint(compiled):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 only")
    torch.manual_seed(123)
    linear = NVFP4Linear.from_linear(
        torch.nn.Linear(512, 512, device="cuda", dtype=torch.bfloat16), name="mlp.up_proj", seed=42
    )
    model = torch.compile(linear) if compiled else linear
    x = torch.randn(1, 129, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = checkpoint(model, x, use_reentrant=False)
    y.float().square().mean().backward()
    assert y.shape == (1, 129, 512)
    assert y.isfinite().all() and x.grad.isfinite().all() and linear.weight.grad.isfinite().all()


def test_software_rounding_is_unbiased_and_reproducible():
    import triton
    import triton.language as tl

    from prime_rl.trainer.models.kernels.nvfp4.hadamard_utils import _pack_fp4

    @triton.jit
    def pack_kernel(source, destination, seed, offset):
        rows = tl.arange(0, 128)[:, None]
        columns = tl.arange(0, 128)[None, :]
        values = tl.load(source + rows * 128 + columns)
        packed = _pack_fp4(values, 128, 128, True, seed, offset, tl.program_id(0))
        tl.store(destination + rows * 64 + tl.arange(0, 64)[None, :], packed)

    source = torch.full((128, 128), 0.75, device="cuda")
    result = torch.empty((128, 64), dtype=torch.uint8, device="cuda")
    seed = torch.tensor([42], device="cuda", dtype=torch.int64)
    offset = torch.tensor([17], device="cuda", dtype=torch.int64)
    pack_kernel[(1,)](source, result, seed, offset)
    previous = result.clone()
    codes = torch.stack((result & 15, result >> 4)).long()
    assert ((codes == 1) | (codes == 2)).all()
    mean = torch.where(codes == 1, 0.5, 1.0).mean().item()
    assert abs(mean - 0.75) < 0.015
    pack_kernel[(1,)](source, result, seed, offset)
    assert torch.equal(result, previous)
    offset.add_(1)
    pack_kernel[(1,)](source, result, seed, offset)
    assert not torch.equal(result, previous)


def test_bucket_groups_scheduler_and_missing_gradients():
    torch.manual_seed(4)
    params = [torch.nn.Parameter(torch.randn(128, device="cuda")) for _ in range(3)]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in params]

    def make_optimizer(ps):
        return torch.optim.AdamW([{"params": ps[:1], "lr": 0.01}, {"params": ps[1:], "lr": 0.02}], foreach=False)

    plain = make_optimizer(reference)
    wrapped = CPUOffloadOptimizer(make_optimizer(params), bucket_bytes=1032)
    a = torch.optim.lr_scheduler.StepLR(plain, step_size=1, gamma=0.5)
    b = torch.optim.lr_scheduler.StepLR(wrapped.base_optimizer, step_size=1, gamma=0.5)
    original = [g["params"] for g in wrapped.param_groups]
    for step in range(3):
        for i, (p, ref) in enumerate(zip(params, reference)):
            grad = torch.randn_like(p) if i != step else None
            p.grad = grad
            ref.grad = grad.clone() if grad is not None else None
        plain.step()
        wrapped.step()
        a.step()
        b.step()
        assert all(torch.equal(p, ref) for p, ref in zip(params, reference))
        assert all(g["params"] is old for g, old in zip(wrapped.param_groups, original))
    with pytest.raises(ValueError, match="exceeds bucket_bytes"):
        CPUOffloadOptimizer(make_optimizer(params), bucket_bytes=1)
