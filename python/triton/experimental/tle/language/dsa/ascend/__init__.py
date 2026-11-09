# Ascend DSA vendor namespace (ported to 3.6).

from .core import (
    UB, L1, L0A, L0B, L0C, PIPE, SyncSpec,
    sub_vec_id, sub_vec_num,
    sync_block_set, sync_block_wait, sync_block_all,
    compile_hint, raw, multibuffer,
)
try:
    from triton.language.extra.cann.extension.aux_ops import parallel
except Exception:
    parallel = None
from .pipe import (
    PipeScheduler, pipe_reader, pipe_slot, pipe_value, pipe_wait_result,
    pipe_writer, pipe_scheduler, pipe, run_pipe, reset_pipe_event_allocator,
)
from .distributed import swizzle2d_Nz, gemm_swizzle2d_Nz

__all__ = [
    "UB", "L1", "L0A", "L0B", "L0C", "PIPE", "SyncSpec",
    "sub_vec_id", "sub_vec_num", "sync_block_set", "sync_block_wait", "sync_block_all",
    "compile_hint", "raw", "multibuffer", "parallel",
    "PipeScheduler", "pipe_reader", "pipe_slot", "pipe_value", "pipe_wait_result",
    "pipe_writer", "pipe_scheduler", "pipe", "run_pipe", "reset_pipe_event_allocator",
    "swizzle2d_Nz", "gemm_swizzle2d_Nz",
]
