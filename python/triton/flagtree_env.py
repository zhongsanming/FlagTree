"""FlagTree-owned environment toggles for compiler passes.

Kept out of ``triton.knobs`` (upstream) so upstream rebases of that file do not
conflict. The C++ side declares the same names in
``include/flagtree/Common/EnvVars.h``, which ``get_cache_invalidating_env_vars``
merges into the upstream cache-invalidating list, so flipping any of these
still invalidates the JIT cache.
"""

import os


def _flagtree_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "on", "yes")


def is_lane_vectorize_disabled() -> bool:
    """Whether the ``triton-lane-vectorize`` TTIR pass should be skipped.

    Mirrors the toggle the Ascend backend has always had, so that every backend
    can turn the pass off the same way. The pass is enabled by default; set
    ``TRITON_DISABLE_LANE_VECTORIZE=1`` to skip it.
    """
    return _flagtree_bool("TRITON_DISABLE_LANE_VECTORIZE", False)


def is_lane_vectorize_concat_allowed() -> bool:
    """Whether the pass may emit the generic ``tensor.concat`` packing path.

    Off by default: the concat materializes its operands (an allocation the
    bundled Ascend BiSheng/HIVM pipeline cannot lower). Only the coalesced
    ``tensor.extract_slice`` path packs unless this is enabled.
    """
    return _flagtree_bool("TRITON_LANE_VECTORIZE_ALLOW_CONCAT", False)


def are_lane_vectorize_address_cones_allowed() -> bool:
    """Whether block mode may pack integer/index cones that feed addresses.

    Off by default: packing address arithmetic and unpacking it again is wrong
    for strided/non-contiguous accesses.
    """
    return _flagtree_bool("TRITON_LANE_VECTORIZE_ALLOW_ADDRESS_CONES", False)


def is_mlir_print_op_generic() -> bool:
    """Whether dumped stage IR is serialized in the generic op form.

    Equivalent to MLIR's ``--mlir-print-op-generic``. Off by default; when set,
    ``python/triton/runtime/cache.py`` prints modules/ops with
    ``get_asm(print_generic_op_form=True)``, giving canonical/diffable IR dumps.
    """
    return _flagtree_bool("TRITON_MLIR_PRINT_OP_GENERIC", False)
