# NVFP4 training on SM120

NVFP4 is experimental. Dense MLP projections use native FP4 GEMMs in forward,
input-gradient and weight-gradient computation. BF16 parameters, gradients and
Adam moments retain their configured dtypes. Attention and the LM head retain
their existing computation. Inference KV cache precision is independent.

```toml
[trainer.model]
attn = "flash_attention_2"
cp = 2
offload = "optimizer"
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"

[trainer.model.quantization]
type = "nvfp4"
seed = 42

[trainer.optim]
offload_bucket_bytes = 4294967296
```

The NVFP4 path requires SM120. Software stochastic rounding uses Philox random
numbers to choose neighboring E2M1 values with distance-proportional probability,
without the unsupported `cvt.rs` instruction. RHT bases and fixed rounding keys
are derived from the seed and module name; each backward consumes fresh offsets
from PyTorch's CUDA RNG. Resume with the same quantization seed and module names.
Quantization metadata is runtime-only and does not change checkpoint weight keys
or the BF16 weights broadcast to inference.

Only language-model `mlp.gate_proj`, `up_proj` and `down_proj` linears are selected
by default. K/N dimensions must be divisible by 128. Token rows are padded and
trimmed automatically. Empty selection is an error. LoRA is unsupported.
Keep activation checkpointing and activation offloading enabled. Initial Triton
compilation/autotuning can take several minutes for each distinct matrix shape.

`offload_bucket_bytes` is an optional AdamW-only setting and requires
`model.offload = "optimizer"`. It bounds resident moment tensors during updates,
including their first initialization. A parameter whose local moment storage
exceeds the budget fails at optimizer construction. Parameters, gradients,
communication buffers and optimizer temporary workspaces are additional memory.
Gradient clipping and the scheduler run once per complete training step.
Checkpoint templates and saved optimizer states stay on CPU. Bucket updates do
not support closures. Omit the field to use the original offload behavior.

For HERO's four-card experiment:

```sh
CUDA_VISIBLE_DEVICES=2,3,0,1 uv run rl @ ../configs/exp/4_rl/6000d/qwen3.8-27b.toml
```

The launcher assigns inference GPUs first, giving physical GPUs 2/3 to inference
and 0/1 to training. All existing environment and sampling settings are retained.
Use a separate output directory for each comparison. Short smoke tests establish
execution compatibility, not reward convergence or an end-to-end speedup.

The kernels are derived from TorchAO 0.18.0, commit `5f2baf9`, under BSD-3-Clause;
see `src/prime_rl/trainer/models/kernels/nvfp4/LICENSE`. Changes include software
stochastic rounding and project-local imports. The installed environment is not
patched.

## Performance validation

Stochastic quantization uses a small 16x128 tile on SM120, retaining the RHT,
block scales and the original 128x128 tile's per-element Philox counters.
The non-stochastic forward path retains its TMA kernel. Both paths preserve
the packed tensor layout and BF16 parameter/checkpoint interfaces.

Compare outputs and warmed timing with the original reference:

```sh
uv run python tools/benchmark_nvfp4_sr.py 8192 17408 --iterations 20 --output /tmp/nvfp4-sr.json
```

Profile a warmed step of a standalone trainer fake-data config:

```sh
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc-per-node=2 tools/profile_training_step.py /tmp/trainer.json /tmp/nvfp4-profile --step 3
uv run python tools/analyze_training_trace.py /tmp/nvfp4-profile/trace_rank_0.json.gz --output /tmp/nvfp4-summary.json
```

The config must enable data.fake, set max_steps to at least the selected step,
and leave trace_path unset. The driver instruments runtime dispatch in its own
process, including compiled calls and checkpoint recompute; it does not change
model precision, checkpoint policy or offload settings. It uses private PyTorch
2.13 profiler/checkpoint APIs.

CPU scope durations include enqueue, synchronization and waiting. GPU activity
durations may overlap across streams. Recompute is nested in backward and
includes some quantization and communication, so its duration must not be added
to those categories. NCCL kernel time includes waiting and is not pure transfer
time.

## Full 27B fake-data validation

`configs/debug/fake/nvfp4-27b.toml` loads the full checkpoint at
`/home/shanxianzheng/Hugging_Face/Models/Qwen/Qwen3.8-27B` with 32K token rows,
CP=2, BF16 weights/reductions, full activation checkpointing, compilation and
4 GiB optimizer-state buckets. The final step is checkpointed.

On this RTX 6000D host, default NCCL P2P stalls during communicator setup;
the two-rank all-reduce succeeds with `NCCL_P2P_DISABLE=1`.
Export these settings before both trainer-only launches:

```sh
export CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_max_round_threshold_mb:16
uv run --frozen --no-sync torchrun --standalone --nproc-per-node=2 -m prime_rl.trainer.rl.train @ configs/debug/fake/nvfp4-27b.toml @ configs/debug/fake/nvfp4-27b-offload.toml
uv run --frozen --no-sync torchrun --standalone --nproc-per-node=2 -m prime_rl.trainer.rl.train @ configs/debug/fake/nvfp4-27b.toml @ configs/debug/fake/nvfp4-27b-offload.toml --max-steps 7 --resume.step 5
```

When sharing the sibling HERO environment without syncing, also export
`UV_PROJECT_ENVIRONMENT=../prime-rl/.venv` and
`PYTHONPATH="$PWD/src:$PWD/packages/prime-rl-configs/src"` from this worktree.
Verify imported source paths before launching. Random fake-data losses do not
establish reward quality or convergence.

The base config retains the experiment setting `ac_offloading = "None"`;
its first 32K backward exhausted the 85,651 MiB devices. Compose the offload
overlay shown above to enable the default activation offloading and use a
separate output directory. Model, sequence length and numerical precision are
unchanged. The default P2P hang and the original OOM logs remain in
`outputs/nvfp4-fake-27b/`.

Bucketed optimizer checkpoints keep moment tensors on CPU. Packed QKV moments
use matching Gloo meshes for checkpoint splitting and reconstruction; their
live CUDA-mesh metadata and update buckets are retained. Checkpoint writes use
serial CPU staging for bucketed optimizers, avoiding asynchronous GPU-to-CPU
staging failures after resume. Canonical weight names and mapping.json remain
available.

The distributed regression can be run without a model training job:

```sh
uv run --frozen --no-sync torchrun --standalone --nproc-per-node=2 -m pytest tests/unit/train/test_nvfp4_training.py -k fused_offloaded_state -q
```
