import hashlib
import re

import torch
from torch import nn
from torch.nn import functional as F

from prime_rl.configs.trainer import NVFP4Config
from prime_rl.utils.logger import get_logger


class NVFP4Linear(nn.Linear):
    """BF16 parameters with native NVFP4 forward, dgrad, and wgrad on SM120."""

    @classmethod
    def from_linear(cls, module: nn.Linear, *, name: str, seed: int) -> "NVFP4Linear":
        with torch.device("meta"):
            new = cls(module.in_features, module.out_features, bias=module.bias is not None)
        new.weight = module.weight
        new.bias = module.bias
        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
        new.sign_vector = tuple(1 if (digest[i // 8] >> (i % 8)) & 1 else -1 for i in range(16))
        new.sr_seed = int.from_bytes(digest[2:10], "little") & (2**63 - 1)
        new.register_buffer(
            "_sr_seed", torch.empty(1, dtype=torch.int64, device=module.weight.device), persistent=False
        )
        if module.weight.device.type != "meta":
            new.reset_runtime()
        return new

    def reset_runtime(self) -> None:
        from prime_rl.trainer.models.kernels.nvfp4.hadamard_utils import prepare_for_cuda_graph

        self._sr_seed.fill_(self.sr_seed)
        prepare_for_cuda_graph(self._sr_seed.device, sign_vectors=(self.sign_vector,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from prime_rl.trainer.models.kernels.nvfp4.nvfp4_linear import nvfp4_linear

        shape = input.shape
        flat = input.reshape(-1, self.in_features)
        padding = (-flat.shape[0]) % 128
        if padding:
            flat = F.pad(flat, (0, 0, 0, padding))
        output = nvfp4_linear(flat, self.weight, self.bias, sign_vector=self.sign_vector, sr_seed=self._sr_seed)
        return output[: input.numel() // self.in_features].reshape(*shape[:-1], self.out_features)


def initialize_nvfp4_runtime(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, NVFP4Linear):
            module.reset_runtime()


def replace_linear_with_nvfp4_linear(model: nn.Module, config: NVFP4Config) -> None:
    include = [re.compile(pattern) for pattern in config.include_patterns]
    ignore = [re.compile(pattern) for pattern in config.ignore_patterns + config.extra_ignore_patterns]
    replaced, skipped, unaligned = [], [], []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(pattern.search(name) for pattern in include) or any(pattern.search(name) for pattern in ignore):
            skipped.append(name)
            continue
        if module.in_features % 128 or module.out_features % 128:
            unaligned.append(name)
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, NVFP4Linear.from_linear(module, name=name, seed=config.seed))
        replaced.append(name)
    get_logger().info(
        f"NVFP4 software SR: replaced={len(replaced)}, skipped={len(skipped)}, unaligned={unaligned}; "
        f"seed={config.seed}, first replaced={replaced[:3]}"
    )
    if not replaced:
        raise ValueError("NVFP4 selection did not match any 128-aligned dense MLP projections")
