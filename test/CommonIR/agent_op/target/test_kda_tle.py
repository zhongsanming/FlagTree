"""Numerical check: v1's Triton kernel against a serial PyTorch reference.

Pytest-compatible, but also runnable directly (v0 and v1 live in this
same directory):

    python3 test_kda_tle.py

Only the prefill H=4/8 path is covered, because that is the only regime v0's
optimized kernel (and therefore v1) is dispatched to; everything else falls
through to v0's generic kernel, which these tests do not re-validate.

Tolerances are the harness's (``RTOL``/``ATOL`` in bench_kda_tle.py).  v1 is
explicitly *not* bit-exact with upstream — see the module docstring of
glm_kda_varlen_recurrent_npu_v1.py — so these are tolerance assertions, not
equality assertions.  The reference is a straight per-token fp32 loop, so it is
independent of both Triton kernels.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

try:  # registers the "npu" backend; absent on a CPU-only box
    import torch_npu  # noqa: F401
    _HAS_NPU = torch.npu.is_available()
except Exception:  # pragma: no cover
    _HAS_NPU = False

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from bench_kda_tle import (  # noqa: E402
    ATOL,
    RTOL,
    arms,
    clone_state,
    load_modules,
    make_case,
    torch_reference,
)

pytestmark = pytest.mark.skipif(not _HAS_NPU, reason="requires an Ascend NPU")

T_CASES = (64, 128, 256)
H_CASES = (4, 8)


@pytest.fixture(scope="module")
def mods():
    v0, v1 = load_modules(os.environ.get("GLM53_KDA_V0_DIR"))
    torch.npu.set_device(os.environ.get("KDA_DEVICE", "npu:0"))
    return v0, v1


@pytest.mark.parametrize("heads", H_CASES)
@pytest.mark.parametrize("tokens", T_CASES)
def test_v1_matches_reference(mods, heads, tokens):
    _, v1 = mods
    case = make_case(heads, tokens, "npu:0")
    ref_out, ref_state = torch_reference(case)

    state = clone_state(case["state"])
    out = v1.call_v1(initial_state_source=state, **arms(case))
    torch.npu.synchronize()

    out_err = (out.float() - ref_out).abs().max().item()
    state_err = (state.float() - ref_state).abs().max().item()
    assert torch.allclose(out.float(), ref_out, rtol=RTOL, atol=ATOL), (
        f"H={heads} T={tokens}: output differs, max |d| = {out_err:.3e}"
    )
    assert torch.allclose(state.float(), ref_state, rtol=RTOL, atol=ATOL), (
        f"H={heads} T={tokens}: state differs, max |d| = {state_err:.3e}"
    )


@pytest.mark.parametrize("heads", (4,))
def test_v0_v1_agree(mods, heads):
    """v1 must not be worse than v0 relative to the same reference."""
    v0, v1 = mods
    case = make_case(heads, 128, "npu:0")
    ref_out, _ = torch_reference(case)
    kw = arms(case)

    outs = {}
    for name, fn in (("v0", v0.glm_kda_varlen_recurrent_npu),
                     ("v1", v1.call_v1)):
        state = clone_state(case["state"])
        outs[name] = fn(initial_state_source=state, **kw).float()
        torch.npu.synchronize()

    e0 = (outs["v0"] - ref_out).abs().max().item()
    e1 = (outs["v1"] - ref_out).abs().max().item()
    # v1 may differ from v0 (different math order) but must stay within the
    # same tolerance envelope; a regression shows up as e1 >> e0.
    assert e1 <= max(e0 * 4.0, ATOL * 4), f"v0 {e0:.3e} vs v1 {e1:.3e}"


@pytest.mark.parametrize("heads", H_CASES)
def test_state_slots_untouched(mods, heads):
    """Only the indexed slot is written; the rest of the pool is a live cache."""
    _, v1 = mods
    case = make_case(heads, 128, "npu:0")
    before = clone_state(case["state"]).float()
    state = clone_state(case["state"])
    v1.call_v1(initial_state_source=state, **arms(case))
    torch.npu.synchronize()
    after = state.float()
    assert torch.equal(after[0], before[0]), "padding row 0 must never be written"
    assert not torch.equal(after[1], before[1]), "the indexed slot must be updated"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
