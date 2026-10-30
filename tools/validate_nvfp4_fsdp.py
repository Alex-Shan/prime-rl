import argparse
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer

parser = argparse.ArgumentParser()
parser.add_argument("--nvfp4", action="store_true")
args = parser.parse_args()

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (2,))
torch.manual_seed(42)
model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.Linear(128, 128)).cuda().bfloat16()
if args.nvfp4:
    from prime_rl.trainer.models.layers.nvfp4_linear import NVFP4Linear

    for name, module in list(model.named_children()):
        setattr(model, name, NVFP4Linear.from_linear(module, name=name, seed=42))
fully_shard(model, mesh=mesh)
optimizer = CPUOffloadOptimizer(torch.optim.AdamW(model.parameters(), lr=1e-3), bucket_bytes=40000)
for i in range(5):
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    model(x).float().square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
optimizer.prepare_checkpoint_save()
ms, osd = get_state_dict(model, optimizer.base_optimizer)
assert all(
    v.to_local().device.type == "cpu" if hasattr(v, "to_local") else v.device.type == "cpu"
    for s in optimizer.state.values()
    for v in s.values()
)
path = "/tmp/hero-bucket-dcp-smoke-nvfp4" if args.nvfp4 else "/tmp/hero-bucket-dcp-smoke"
dcp.save({"model": ms, "optimizer": osd}, checkpoint_id=path)
expected = [p.to_local().clone() for p in model.parameters()]
with torch.no_grad():
    for p in model.parameters():
        p.zero_()
optimizer.state.clear()
optimizer.prepare_checkpoint_save()
ms, osd = get_state_dict(model, optimizer.base_optimizer)
dcp.load({"model": ms, "optimizer": osd}, checkpoint_id=path)
set_model_state_dict(model, ms)
optimizer.finish_checkpoint_load()
assert all(torch.equal(a, p.to_local()) for a, p in zip(expected, model.parameters()))
model(x).float().square().mean().backward()
optimizer.step()
print(
    f"PASS 2-rank {'NVFP4' if args.nvfp4 else 'BF16'} FSDP 5 updates + CPU DCP save/load + resumed update", flush=True
)
dist.destroy_process_group()

if args.nvfp4:
    from pathlib import Path

    from prime_rl.configs.trainer import CheckpointConfig
    from prime_rl.trainer.ckpt import CheckpointManager, Progress

    dist.init_process_group("nccl")
    manager = CheckpointManager(Path("/tmp/hero-nvfp4-rng-test"), CheckpointConfig())
    checkpoint_path = Path("/tmp/hero-nvfp4-rng-test/state")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.base_optimizer, lambda _: 1.0)
    manager.save_to_path(checkpoint_path, model, [optimizer], scheduler, Progress())
    expected_cpu = torch.rand(16)
    expected_cuda = torch.rand(16, device="cuda")
    manager.load_from_path(checkpoint_path, model, [optimizer], scheduler, Progress())
    assert torch.equal(expected_cpu, torch.rand(16))
    assert torch.equal(expected_cuda, torch.rand(16, device="cuda"))
    print("PASS NVFP4 CheckpointManager RNG save/restore", flush=True)
    dist.destroy_process_group()
