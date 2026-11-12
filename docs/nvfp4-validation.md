# HERO NVFP4 integration validation

## Branch analysis

- Host: shanxianzheng@219.216.80.23; worktree: /home/shanxianzheng/Code/Paper_HERO/HERO/prime-rl-nvfp4.
- Common ancestor: 562b09bc. HERO d92c0351 has 77 unique commits; NVFP4 2263581f has 5.
- HERO supplies the current shared training/pretraining loop, VLM interfaces, environment worker pools, live configuration, checkpoint mapping and optimizer resume fixes.
- NVFP4 supplies SM120 dense MLP kernels, seeded software stochastic rounding, bounded AdamW state buckets, GPU checks and profiling tools.
- Merge commit: 55d61064. Both original tips are ancestors. The import conflict retains NGramTable and NVFP4 runtime imports. Dependency manifests and submodule pins follow HERO.
- Git fetch timed out; the merge uses the host's existing branch refs, not a verified newer upstream tip.

## Integration fixes

CPU-resident packed optimizer tensors use matching explicit Gloo meshes for canonical checkpoint splitting and reconstruction. Live optimizer mesh metadata and bounded GPU buckets remain intact. Bucketed checkpoint writes use synchronous CPU staging; the upstream asynchronous staging path failed after resume. Model weight keys, mapping.json, BF16 dtypes and the NVFP4 configuration interface are retained.

## Full model run

Model: /home/shanxianzheng/Hugging_Face/Models/Qwen/Qwen3.8-27B. The checkpoint declares qwen3_5; the full 64-layer model is loaded, with 192 MLP projections replaced by NVFP4. GPU 0/1, CP=2, sequence length 32768, fake batch size 2, full activation checkpointing, compile, BF16 weights/reductions, AdamW lr=3e-5, 4 GiB state buckets.

The original experiment disables activation offloading and fails with OOM during the first backward. The explicit nvfp4-27b-offload.toml overlay enables the repository default activation offloading without changing sequence length, model size or precision. NCCL_P2P_DISABLE=1 and NCCL_SOCKET_IFNAME=lo resolve this host's reproduced communicator hang. Commands and environment setup are in nvfp4-sm120.md.

Five steps finish, step_5 saves successfully, then two additional updates resume from that checkpoint and step_7 saves successfully. DCP metadata includes optimizer/scheduler/progress; mapping.json and both rank RNG files are present. Saved counters equal 327680 tokens/10 samples at step 5 and 458752 tokens/14 samples at step 7. CUDA RNG states advance on both ranks. Loss and gradient norms are finite, with nonzero gradients at all seven steps.

| Step | Seconds | Tokens/s | Peak GiB | Loss | Grad norm |
|---|---:|---:|---:|---:|---:|
| 1 | 54.36 | 1206 | 69.603 | -9.98362e-08 | 2.19345e-05 |
| 2 | 46.23 | 1418 | 69.603 | -3.02003e-09 | 2.21729e-05 |
| 3 | 46.20 | 1419 | 69.603 | -4.75631e-08 | 2.95639e-05 |
| 4 | 46.22 | 1418 | 69.603 | 5.60682e-08 | 4.81606e-05 |
| 5 | 46.23 | 1418 | 69.603 | 4.40741e-08 | 2.09808e-05 |
| 6 | 47.89 | 1369 | 69.609 | -2.91024e-07 | 3.36170e-05 |
| 7 | 46.19 | 1419 | 69.609 | -1.27439e-07 | 3.17097e-05 |

## Checks and artifacts

- Focused configuration tests including the new standalone config: 5 passed.
- Existing fusion tests: 5 passed.
- Distributed CPU offload fusion regression: 1 passed on each of 2 ranks.
- Full configuration suite: merged branch 174 passed/101 failed; unchanged HERO baseline 171 passed/101 failed. Failure identifiers match exactly. The environment lacks some external taskset plugins (including scaleswe) and has additional existing validation failures; the suite is not fully green.
- Ruff checks and git diff --check pass.
- Metrics: outputs/nvfp4-fake-27b-offload/monitors/file/metrics.jsonl; aggregated summary.json is alongside launch.log and resume.log.
- Checkpoints: outputs/nvfp4-fake-27b-offload/checkpoints/step_{5,7}/trainer (approximately 152 GiB each).
- Preserved failures: outputs/nvfp4-fake-27b/, outputs/nvfp4-fake-27b-offload-checkpoint-failed/, resume-async-staging-failed.log and failed-step7 under the successful run directory.
- Original staged deps/verifiers and outer HERO configuration edits are preserved. No branches are pushed and no formal 500-step training starts.

These tests establish execution and checkpoint compatibility, not reward convergence, bitwise equivalence with an uninterrupted fake-data stream, or a measured speedup against BF16. The fake loader starts its batch-seed sequence again on resume.
