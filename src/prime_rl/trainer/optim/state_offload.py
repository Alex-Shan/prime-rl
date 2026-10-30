import copy

import torch
from torch.distributed.tensor import DTensor
from torch.optim import AdamW, Optimizer

from prime_rl.trainer.optim.base import OffloadOptimizer


class CPUOffloadOptimizer(OffloadOptimizer):
    """Wraps an optimizer to keep states on CPU, moving to GPU only for step().

    Unlike FSDP's CPUOffload which offloads weights too, this keeps weights on GPU.
    With activation checkpointing, activations and optimizer states are never on GPU
    at the same time: peak memory becomes max(activations, opt_states) instead of sum.
    """

    def __init__(self, optimizer: Optimizer, pin_memory: bool = True, *, bucket_bytes: int | None = None):
        self.bucket_bytes = bucket_bytes
        if bucket_bytes is not None and (bucket_bytes < 1 or not isinstance(optimizer, AdamW)):
            raise ValueError("Bucket offload requires AdamW and a positive bucket_bytes")
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._initialized = False
        self._cpu_buffers: dict[tuple[torch.Tensor, str], torch.Tensor] = {}
        if bucket_bytes is not None:
            list(self._buckets())

    def _offload_to_cpu(self, param: torch.Tensor, key: str, src: torch.Tensor) -> torch.Tensor:
        if not self.pin_memory:
            return src.to("cpu", non_blocking=True)
        buffer = self._cpu_buffers.get((param, key))
        if buffer is None or buffer.shape != src.shape or buffer.dtype != src.dtype:
            buffer = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
            self._cpu_buffers[(param, key)] = buffer
        if buffer is not src:
            buffer.copy_(src, non_blocking=True)
        return buffer

    def _move_states(self, device: str, params=None):
        """Move optimizer states to CPU or back to GPU (matching each parameter's device)."""
        for param in self.optimizer.state if params is None else params:
            state = self.optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, DTensor):
                    local_tensor = value._local_tensor
                    if device == "cpu":
                        new_local = self._offload_to_cpu(param, key, local_tensor)
                    else:
                        new_local = local_tensor.to(device, non_blocking=True)
                    new_dtensor = copy.copy(value)
                    new_dtensor._local_tensor = new_local
                    state[key] = new_dtensor
                elif isinstance(value, torch.Tensor):
                    if device == "cpu":
                        state[key] = self._offload_to_cpu(param, key, value)
                    else:
                        state[key] = value.to(device, non_blocking=True)
        if device == "cpu" and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def _buckets(self):
        bucket, size = [], 0
        for group in self.param_groups:
            for param in group["params"]:
                local = param.to_local() if isinstance(param, DTensor) else param
                cost = local.numel() * local.element_size() * (3 if group.get("amsgrad") else 2) + 8
                if cost > self.bucket_bytes:
                    raise ValueError(
                        f"Adam parameter state requires {cost} bytes, exceeds bucket_bytes={self.bucket_bytes}"
                    )
                if bucket and size + cost > self.bucket_bytes:
                    yield bucket
                    bucket, size = [], 0
                bucket.append(param)
                size += cost
        if bucket:
            yield bucket

    def _bucket_step(self):
        groups = self.optimizer.param_groups
        original = [group["params"] for group in groups]
        buckets = list(self._buckets())
        try:
            for bucket in buckets:
                selected = {id(param) for param in bucket}
                for group, params in zip(groups, original):
                    group["params"] = [param for param in params if id(param) in selected]
                existing = [param for param in bucket if param in self.optimizer.state]
                self._move_states("cuda", existing)
                try:
                    self.optimizer.step()
                finally:
                    self._move_states("cpu", [p for p in bucket if p in self.optimizer.state])
        finally:
            for group, params in zip(groups, original):
                group["params"] = params
        self._initialized = True

    def step(self, closure=None):
        if self.bucket_bytes is not None:
            if closure is not None:
                raise ValueError("Bucket offload does not support closures")
            return self._bucket_step()

        # First step initializes states on GPU - offload after
        if not self._initialized:
            result = self.optimizer.step(closure)
            self._move_states("cpu")
            self._initialized = True
            return result

        # Move states to GPU
        self._move_states("cuda")

        # Run optimizer step
        result = self.optimizer.step(closure)

        # Move states back to CPU
        self._move_states("cpu")

        return result

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        if self.bucket_bytes is not None:
            return self.optimizer.state_dict()
        # Move to GPU temporarily for consistent state dict
        if self._initialized:
            self._move_states("cuda")
            torch.cuda.synchronize()
        state_dict = self.optimizer.state_dict()
        if self._initialized:
            self._move_states("cpu")
        return state_dict

    def load_state_dict(self, state_dict):
        if self.bucket_bytes is not None:
            saved_groups = state_dict["param_groups"]
            if len(saved_groups) != len(self.param_groups) or any(
                len(saved["params"]) != len(group["params"]) for saved, group in zip(saved_groups, self.param_groups)
            ):
                raise ValueError("Optimizer parameter groups do not match checkpoint")
            mapping = {}
            for saved, group in zip(saved_groups, self.param_groups):
                mapping.update(zip(saved["params"], group["params"]))
                group.update({key: value for key, value in saved.items() if key != "params"})
            self.optimizer.state.clear()
            for index, state in state_dict["state"].items():
                self.optimizer.state[mapping[index]] = state.copy()
            self._move_states("cpu")
            self._initialized = True
            return
        self.optimizer.load_state_dict(state_dict)
        self._move_states("cpu")
        self._initialized = True

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    @property
    def state(self):
        return self.optimizer.state

    @property
    def base_optimizer(self) -> Optimizer:
        return self.optimizer

    def checkpoint_optimizer(self) -> Optimizer:
        return self.optimizer

    def prepare_checkpoint_save(self) -> None:
        if self.bucket_bytes is not None:
            # DCP initializes an empty optimizer by running a step. Materialize a CPU
            # template first so checkpoint loading never initializes all states on GPU.
            for group in self.param_groups:
                for param in group["params"]:
                    if self.optimizer.state[param]:
                        continue
                    local = param.to_local() if isinstance(param, DTensor) else param
                    state = self.optimizer.state[param]
                    state["step"] = torch.tensor(0.0)
                    for key in ["exp_avg", "exp_avg_sq"] + (["max_exp_avg_sq"] if group.get("amsgrad") else []):
                        value = torch.zeros(local.shape, dtype=local.dtype, device="cpu", pin_memory=self.pin_memory)
                        if isinstance(param, DTensor):
                            wrapped = copy.copy(param.detach())
                            wrapped._local_tensor = value
                            value = wrapped
                        state[key] = value
            return
        if self._initialized:
            self._move_states("cuda")
            torch.cuda.synchronize()

    def finish_checkpoint_save(self) -> None:
        self._move_states("cpu")

    def finish_checkpoint_load(self) -> None:
        self._initialized = True
