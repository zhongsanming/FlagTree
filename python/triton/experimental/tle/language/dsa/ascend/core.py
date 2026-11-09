# Ascend DSA vendor primitives for TLE (3.6 port).
# Constants/helpers are backed by triton.language.extra.cann.extension where available.

from dataclasses import dataclass

try:
    import triton.language.extra.cann.extension as _cann_ext
    from triton.language.extra.cann.extension.core import ascend_address_space, sub_vec_id, sub_vec_num, sync_block_set, sync_block_wait, sync_block_all
    _CANN = True
except Exception:
    _cann_ext = None
    ascend_address_space = None
    sub_vec_id = sub_vec_num = sync_block_set = sync_block_wait = sync_block_all = None
    _CANN = False

try:
    from triton.language.extra.cann.extension import compile_hint, custom as raw, multibuffer
except Exception:
    compile_hint = raw = multibuffer = None

if _CANN and ascend_address_space is not None:
    UB = ascend_address_space.UB
    L1 = ascend_address_space.L1
    L0A = ascend_address_space.L0A
    L0B = ascend_address_space.L0B
    L0C = ascend_address_space.L0C
    PIPE = _cann_ext.PIPE
else:
    # Fallback string tokens for environments without cann extension resolved;
    # backend lowering maps these to Ascend memory spaces.
    UB, L1, L0A, L0B, L0C = "ub", "l1", "l0a", "l0b", "l0c"
    PIPE = None


@dataclass(frozen=True)
class SyncSpec:
    """Cross-core event direction used by the Ascend pipe scheduler."""
    __triton_compile_time_value__ = True
    backend = "ascend"

    sender: str
    receiver: str
    sender_pipe: object
    receiver_pipe: object

    def __post_init__(self):
        if self.sender not in ("cube", "vector"):
            raise ValueError(f"sender must be 'cube' or 'vector', got {self.sender!r}")
        if self.receiver not in ("cube", "vector"):
            raise ValueError(f"receiver must be 'cube' or 'vector', got {self.receiver!r}")
        if self.sender == self.receiver:
            raise ValueError("sender and receiver must be different")
