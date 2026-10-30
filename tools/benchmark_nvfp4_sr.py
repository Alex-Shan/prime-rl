"""Compare SM120 stochastic quantization with the original 128x128 reference."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from prime_rl.trainer.models.kernels.nvfp4.hadamard_amax_triton import triton_rht_amax
from prime_rl.trainer.models.kernels.nvfp4.hadamard_quantize_row_col_triton import (
    _hadamard_quantize_row_col_kernel,
    triton_rht_quantize_row_col,
)
from prime_rl.trainer.models.kernels.nvfp4.hadamard_utils import (
    _device_key,
    get_rht_matrix,
    prepare_for_cuda_graph,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=int)
    parser.add_argument("columns", type=int)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 128 or args.columns < 128 or args.rows % 128 or args.columns % 128:
        parser.error("rows and columns must be positive multiples of 128")
    if args.iterations < 1:
        parser.error("iterations must be positive")
    torch.manual_seed(42)
    m, n = args.rows, args.columns
    signs = tuple(1 if i % 3 else -1 for i in range(16))
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.01
    seed = torch.tensor([12345], device="cuda", dtype=torch.int64)
    row_seed = seed ^ 1
    col_offset = torch.tensor([71], device="cuda", dtype=torch.int64)
    row_offset = col_offset + 2
    col_amax, row_amax = triton_rht_amax(x, sign_vector=list(signs))
    hadamard = get_rht_matrix(signs, _device_key(x.device), torch.bfloat16, 16)
    workspace = prepare_for_cuda_graph(x.device, sign_vectors=(signs,))
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    reference = (
        torch.empty(n, m // 2, device="cuda", dtype=torch.uint8),
        torch.empty(n // 128, m // 64, 32, 16, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(m, n // 2, device="cuda", dtype=torch.uint8),
        torch.empty(m // 128, n // 64, 32, 16, device="cuda", dtype=torch.float8_e4m3fn),
    )

    def baseline():
        triton.set_allocator(lambda size, align, stream: workspace[: max(size, 1)])
        try:
            _hadamard_quantize_row_col_kernel[(sms,)](
                x,
                hadamard,
                col_amax,
                row_amax,
                *reference,
                seed,
                col_offset,
                row_seed,
                row_offset,
                m,
                n,
                GROUP_SIZE_N=8,
                NUM_SMS=sms,
                STOCHASTIC_ROUNDING=True,
            )
        finally:
            triton.set_allocator(None)
        return reference

    def optimized():
        return triton_rht_quantize_row_col(
            x,
            col_amax,
            row_amax,
            list(signs),
            True,
            col_seed_base=seed,
            col_offset_base=col_offset,
            row_seed_base=row_seed,
            row_offset_base=row_offset,
        )

    before, after = baseline(), optimized()
    torch.cuda.synchronize()
    different = [int((a.view(torch.uint8) != b.view(torch.uint8)).sum()) for a, b in zip(before, after)]
    if any(different):
        raise AssertionError(f"FP4 codes/scales differ: {different}")
    result = {"rows": m, "columns": n, "gpu": torch.cuda.get_device_name(), "different_bytes": different}
    for name, function in (("baseline", baseline), ("optimized", optimized)):
        for _ in range(3):
            function()
        samples = []
        for _ in range(args.iterations):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            function()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end))
        result[name + "_median_ms"] = statistics.median(samples)
        result[name + "_samples_ms"] = samples
    result["speedup"] = result["baseline_median_ms"] / result["optimized_median_ms"]
    content = json.dumps(result, indent=2)
    print(content)
    if args.output:
        args.output.write_text(content + "\n")


if __name__ == "__main__":
    main()
