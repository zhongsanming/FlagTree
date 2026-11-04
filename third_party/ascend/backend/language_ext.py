"""Backend-provided language extensions for the Ascend TensorView frontend."""

import triton.language as tl
from triton.language.core import builtin

__all__ = [
    "make_partition_view",
    "make_strided_view",
    "make_gather_scatter_view",
]


@builtin
def make_partition_view(base, shape, strides, tile, padding_value="zero", _semantic=None):
    return _semantic.make_partition_view(base, shape, strides, tile, padding_value)


@builtin
def make_strided_view(base, shape, strides, tile, traversal_strides, padding_value="zero", _semantic=None):
    return _semantic.make_strided_view(base, shape, strides, tile, traversal_strides, padding_value)


@builtin
def make_gather_scatter_view(base, shape, strides, tile, sparse_dim, padding_value="zero", _semantic=None):
    return _semantic.make_gather_scatter_view(base, shape, strides, tile, sparse_dim, padding_value)
