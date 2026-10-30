"""Profile one warmed trainer step without changing model precision or checkpoint policy."""

import argparse
import functools
import json
import os
import threading
from pathlib import Path

import torch
import torch.utils.checkpoint as checkpoint_module
from torch.profiler import ProfilerActivity, profile, record_function

import prime_rl.trainer.rl.train as trainer
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer
from prime_rl.utils.config import cli

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--step", type=int, default=3)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
config = cli(TrainerConfig, args=["@", str(args.config)])
if config.data.fake is None:
    parser.error("profiling driver requires data.fake")
if config.trace_path is not None:
    parser.error("disable trainer trace_path; this driver owns the profiler")
if args.step < 1 or (config.max_steps is not None and args.step > config.max_steps):
    parser.error("selected step must be within max_steps")

rank = int(os.environ.get("RANK", 0))
local = threading.local()
state = {"step": 0, "prof": None, "range": None, "done": False}


def wrap_function(owner, name, label):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapped(*a, **kw):
        if state["prof"] is None:
            return original(*a, **kw)
        with record_function(label(*a, **kw) if callable(label) else label):
            return original(*a, **kw)

    setattr(owner, name, wrapped)


wrap_function(trainer, "forward", "train/forward")
wrap_function(torch.Tensor, "backward", "train/backward")
wrap_function(CPUOffloadOptimizer, "_bucket_step", "train/optimizer")
wrap_function(CPUOffloadOptimizer, "_move_states", lambda self, device, *a, **kw: "offload/states_to_" + device)

original_frame = checkpoint_module._CheckpointFrame.__init__


@functools.wraps(original_frame)
def frame_init(self, recompute_fn, *a, **kw):
    @functools.wraps(recompute_fn)
    def recompute(*ra, **rkw):
        if state["prof"] is None:
            return recompute_fn(*ra, **rkw)
        with record_function("train/recompute"):
            return recompute_fn(*ra, **rkw)

    return original_frame(self, recompute, *a, **kw)


checkpoint_module._CheckpointFrame.__init__ = frame_init

# Annotate actual runtime custom-op dispatches, including compiled execution.
from prime_rl.trainer.models.kernels.nvfp4.hadamard_amax_triton import triton_rht_amax
from prime_rl.trainer.models.kernels.nvfp4.hadamard_quantize_row_col_triton import triton_rht_quantize_row_col
from prime_rl.trainer.models.kernels.nvfp4.nvfp4_linear import _nvfp4_scaled_mm
from prime_rl.trainer.models.kernels.nvfp4.quantize_2d_triton import triton_weight_quantize_2d


def annotate_op(op, kind):
    original = op._init_fn

    def implementation(*a, **kw):
        if state["prof"] is None:
            return original(*a, **kw)
        if kind == "quantize":
            sr = kw.get("stochastic_rounding", a[4] if len(a) > 4 else False)
            local.phase = "backward" if sr else ("recompute" if torch._C._current_graph_task_id() != -1 else "forward")
        phase = getattr(local, "phase", "unassigned")
        with record_function("nvfp4/" + kind + "/" + phase):
            return original(*a, **kw)

    op.register_kernel("cuda", implementation)


for op, kind in [
    (triton_rht_quantize_row_col, "quantize"),
    (triton_rht_amax, "amax"),
    (triton_weight_quantize_2d, "weight_quantize"),
    (_nvfp4_scaled_mm, "gemm"),
]:
    annotate_op(op, kind)

original_wait = trainer.FakeDataLoader.wait_for_batch


@functools.wraps(original_wait)
def wait(self):
    state["step"] += 1
    if state["step"] == args.step:
        torch.cuda.synchronize()
        state["prof"] = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False
        ).__enter__()
        state["range"] = record_function("train/step")
        state["range"].__enter__()
    return original_wait(self)


trainer.FakeDataLoader.wait_for_batch = wait

original_log = trainer.monitors.log


@functools.wraps(original_log)
async def log(metrics, *a, **kw):
    result = await original_log(metrics, *a, **kw)
    if state["prof"] is not None and "time/step" in metrics:
        torch.cuda.synchronize()
        state["range"].__exit__(None, None, None)
        prof = state["prof"]
        prof.__exit__(None, None, None)
        state["prof"] = None
        prof.export_chrome_trace(str(args.output / f"trace_rank_{rank}.json.gz"))
        (args.output / f"metrics_rank_{rank}.json").write_text(json.dumps(metrics))
        state["done"] = True
    return result


trainer.monitors.log = log
trainer.train(config)
assert state["done"], "Selected step was not profiled"
