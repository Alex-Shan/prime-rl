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
CUDA_VISIBLE_DEVICES=2,3,0,1 uv run rl @ ../configs/exp/3_rl/6000d/qwen3.8-27b-nvfp4.toml
```

The launcher assigns inference GPUs first, giving physical GPUs 2/3 to inference
and 0/1 to training. All existing environment and sampling settings are retained.
Use a separate output directory for each comparison. Short smoke tests establish
execution compatibility, not reward convergence or an end-to-end speedup.

The kernels are derived from TorchAO 0.18.0, commit `5f2baf9`, under BSD-3-Clause;
see `src/prime_rl/trainer/models/kernels/nvfp4/LICENSE`. Changes include software
stochastic rounding and project-local imports. The installed environment is not
patched.
