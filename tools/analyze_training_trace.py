import argparse
import collections
import gzip
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("trace", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
reader = gzip.open if args.trace.suffix == ".gz" else open
with reader(args.trace, "rt") as f:
    events = json.load(f)["traceEvents"]
annotations = [e for e in events if e.get("cat") == "user_annotation" and e.get("ph") == "X"]
step = next(e for e in annotations if e["name"] == "train/step")
wall = step["dur"] / 1e6
cpu = collections.defaultdict(float)
for e in annotations:
    cpu[e["name"]] += e["dur"] / 1e6
ranges = collections.defaultdict(list)
for e in annotations:
    if (
        e["name"] == "train/recompute"
        or e["name"].startswith("offload/")
        or e["name"] in ("train/forward", "train/backward", "train/optimizer")
    ):
        ranges[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
for v in ranges.values():
    v.sort()
runtime = {
    e["args"]["correlation"]: e
    for e in events
    if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("ph") == "X" and "correlation" in e.get("args", {})
}
work = collections.defaultdict(float)
nested = collections.defaultdict(float)
kernel_totals = collections.Counter()
bytes_by_kind = collections.Counter()
for e in events:
    cat = e.get("cat")
    if cat not in ("kernel", "gpu_memcpy", "gpu_memset") or e.get("ph") != "X":
        continue
    name = e["name"]
    duration = e["dur"] / 1e6
    host = runtime.get(e.get("args", {}).get("correlation"))
    parents = []
    if host:
        parents = [label for start, end, label in ranges.get(host["tid"], []) if start <= host["ts"] <= end]
    if "train/recompute" in parents:
        nested["recompute_gpu_s"] += duration
    if cat == "kernel":
        kernel_totals[name] += duration
        if "nccl" in name.lower():
            kind = "communication"
        elif "bstensorop" in name or "gemm" in name.lower():
            kind = "gemm"
        elif any(
            x in name
            for x in (
                "hadamard",
                "quantize_2d",
                "quantize_sr",
                "triton_per_fused__to_copy_abs_div_max",
                "triton_red_fused__to_copy_abs_max",
                "randint",
            )
        ):
            kind = "quantization"
        else:
            kind = "other_compute"
    elif cat == "gpu_memcpy":
        if any(p.startswith("offload/") for p in parents):
            kind = "optimizer_state_transfer"
        elif "HtoD" in name or "DtoH" in name:
            kind = "activation_and_other_host_transfer"
        else:
            kind = "device_copy"
        bytes_by_kind[kind] += e.get("args", {}).get("bytes", 0)
    else:
        kind = "memset"
    work[kind] += duration
    if kind == "quantization" and "train/recompute" in parents:
        nested["recompute_quantization_s"] += duration
    if kind == "communication" and "train/recompute" in parents:
        nested["recompute_communication_s"] += duration
sum_gpu = sum(work.values())
summary = {
    "trace": str(args.trace),
    "step_wall_s": wall,
    "cpu_scopes_s": dict(cpu),
    "gpu_activity_s": dict(work),
    "gpu_activity_share_pct": {k: 100 * v / sum_gpu for k, v in work.items()},
    "gpu_activity_sum_s": sum_gpu,
    "nested_gpu_s": dict(nested),
    "transferred_gib": {k: v / 2**30 for k, v in bytes_by_kind.items()},
    "top_gpu_kernels_s": kernel_totals.most_common(15),
    "notes": [
        "GPU activity duration sums can overlap across streams; percentages are shares of summed GPU activity, not wall-clock attribution.",
        "Recompute is nested within backward and includes some quantization/communication; do not add those categories.",
        "CPU scopes include enqueue, synchronization and host work; raw scopes may be nested.",
    ],
}
text = json.dumps(summary, indent=2)
if args.output:
    args.output.write_text(text)
print(text)
