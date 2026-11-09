# Ascend DSA distributed helpers (swizzle). Ported for 3.6 tutorials.

import triton.language as tl
from triton.language.core import builtin


@builtin
def swizzle2d_Nz(
    iter_id,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    N: tl.constexpr = 4,
    _semantic=None,
):
    """N-dimensional column-major swizzle used by allgather+gemm tutorials."""
    cta_per_rank = tl.cdiv(data_row_shape, tile_row_shape) * tl.cdiv(data_col_shape, tile_col_shape)
    local_cta_id = (iter_id // rank_size) % cta_per_rank
    target_rank_id = iter_id % rank_size
    num_row_blocks = tl.cdiv(data_row_shape, tile_row_shape)
    num_col_blocks = tl.cdiv(data_col_shape, tile_col_shape)
    num_local_blocks = local_cta_id % (N * num_row_blocks)
    row_id = num_local_blocks % num_row_blocks
    n_id = num_local_blocks // num_row_blocks
    col_id = (local_cta_id // (N * num_row_blocks)) * N + n_id
    row_tile_shape = tl.minimum(data_row_shape - row_id * tile_row_shape, tile_row_shape)
    col_tile_shape = tl.minimum(data_col_shape - col_id * tile_col_shape, tile_col_shape)
    return row_id, col_id, target_rank_id, row_tile_shape, col_tile_shape


@builtin
def gemm_swizzle2d_Nz(
    iter_id,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    N: tl.constexpr = 4,
    _semantic=None,
):
    """Column-major swizzle for GEMM tiling used by allgather+gemm tutorial."""
    num_row_blocks = tl.cdiv(data_row_shape, tile_row_shape)
    num_col_blocks = tl.cdiv(data_col_shape, tile_col_shape)
    num_local_blocks = iter_id % (N * num_row_blocks)
    row_id = num_local_blocks % num_row_blocks
    n_id = num_local_blocks // num_row_blocks
    col_id = (iter_id // (N * num_row_blocks)) * N + n_id
    return row_id, col_id
