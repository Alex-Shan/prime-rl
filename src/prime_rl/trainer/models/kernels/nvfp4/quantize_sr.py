# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the BSD-3-Clause license; see LICENSE in this directory.
"""Small non-persistent tiles for SM120 RHT and software stochastic rounding."""

import triton
import triton.language as tl

from .hadamard_utils import _nvfp4_quantize, _pack_fp4


@triton.jit
def _scale_offsets(outer, inner_block, inner_size):
    return (
        ((outer // 128) * (inner_size // 64) + inner_block // 4) * 512
        + (outer % 32) * 16
        + ((outer % 128) // 32) * 4
        + inner_block % 4
    )


@triton.jit
def _canonical_indices(m, n, M: tl.constexpr, N: tl.constexpr):
    # Match the original 128x128 grouped tile's Philox counters for every element.
    pm = m // 128
    pn = n // 128
    group = pn // 8
    width = tl.minimum(N // 128 - group * 8, 8)
    tile = group * (8 * (M // 128)) + pm * width + pn % 8
    return tile.to(tl.uint64) * 16384


@triton.jit
def quantize_sr_small_kernel(
    A,
    H,
    COL_AMAX,
    ROW_AMAX,
    COL,
    COL_SF,
    ROW,
    ROW_SF,
    COL_SEED,
    COL_OFFSET,
    ROW_SEED,
    ROW_OFFSET,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 128,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    a = tl.load(A + m[:, None] * N + n[None, :])
    h = tl.load(H + tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])
    transposed = tl.trans(a).reshape(BLOCK_N * BLOCK_M // 16, 16)
    transformed = tl.dot(transposed, h).to(tl.bfloat16)
    col_scale, col_scaled = _nvfp4_quantize(transformed, tl.load(COL_AMAX), BLOCK_N, BLOCK_M)
    col_index = _canonical_indices(m[None, :], n[:, None], M, N) + (n[:, None] % 128) * 128 + m[None, :] % 128
    col_codes = _pack_fp4(col_scaled, BLOCK_N, BLOCK_M, True, COL_SEED, COL_OFFSET, 0, col_index)
    tl.store(
        COL + n[:, None] * (M // 2) + (tl.program_id(0) * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2))[None, :],
        col_codes,
    )
    col_blocks = tl.program_id(0) * (BLOCK_M // 16) + tl.arange(0, BLOCK_M // 16)
    tl.store(COL_SF + _scale_offsets(n[:, None], col_blocks[None, :], M), col_scale)

    row_scale, row_scaled = _nvfp4_quantize(a, tl.load(ROW_AMAX), BLOCK_M, BLOCK_N)
    row_index = _canonical_indices(m[:, None], n[None, :], M, N) + (m[:, None] % 128) * 128 + n[None, :] % 128
    row_codes = _pack_fp4(row_scaled, BLOCK_M, BLOCK_N, True, ROW_SEED, ROW_OFFSET, 0, row_index)
    tl.store(
        ROW + m[:, None] * (N // 2) + (tl.program_id(1) * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2))[None, :],
        row_codes,
    )
    row_blocks = tl.program_id(1) * (BLOCK_N // 16) + tl.arange(0, BLOCK_N // 16)
    tl.store(ROW_SF + _scale_offsets(m[:, None], row_blocks[None, :], N), row_scale)
