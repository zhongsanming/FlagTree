# flagtree tle
"""TsingMicro vendor DSA rand-family tests (tle.dsa.tsingmicro). txda-only."""

import pytest
torch = pytest.importorskip("torch")

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _is_txda():
    try:
        import torch_txda  # noqa: F401
    except ImportError:
        return False
    try:
        target = triton.runtime.driver.active.get_current_target()
        return getattr(target, "backend", None) == "txda"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _is_txda(), reason="TLE DSA rand tests require TsingMicro (txda) backend")


@triton.jit
def _randgen_kernel(seed_val, out_ptr, s0_ptr, s1_ptr, N: tl.constexpr):
    s0 = tl.arange(0, 16).to(tl.int64) * 0x2545F4914F6CDD1D + seed_val
    s1 = tl.arange(0, 16).to(tl.int64) * 0x1E3779B97F4A7C15 + seed_val + 1
    out, s0o, s1o = tle.dsa.tsingmicro.randgen(s0, s1, N)
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, out)
    tl.store(s0_ptr + tl.arange(0, 16), s0o)
    tl.store(s1_ptr + tl.arange(0, 16), s1o)


@triton.jit
def _rand_kernel(seed_val, out_ptr, N: tl.constexpr):
    s0 = tl.arange(0, 16).to(tl.int64) * 0x2545F4914F6CDD1D + seed_val
    s1 = tl.arange(0, 16).to(tl.int64) * 0x1E3779B97F4A7C15 + seed_val + 1
    u, s0o, s1o = tle.dsa.tsingmicro.rand(s0, s1, N)
    tl.store(out_ptr + tl.arange(0, N), u)


@triton.jit
def _randn_kernel(seed_val, out_ptr, N: tl.constexpr):
    s0 = tl.arange(0, 16).to(tl.int64) * 0x2545F4914F6CDD1D + seed_val
    s1 = tl.arange(0, 16).to(tl.int64) * 0x1E3779B97F4A7C15 + seed_val + 1
    n, s0o, s1o = tle.dsa.tsingmicro.randn(s0, s1, N)
    tl.store(out_ptr + tl.arange(0, N), n)


def test_randgen_shape_and_determinism():
    n = 32
    out = torch.empty(n, device="txda", dtype=torch.int64)
    s0o = torch.zeros(16, device="txda", dtype=torch.int64)
    s1o = torch.zeros(16, device="txda", dtype=torch.int64)
    _randgen_kernel[(1, )](42, out, s0o, s1o, N=n)
    out2 = torch.empty_like(out)
    _randgen_kernel[(1, )](42, out2, s0o, s1o, N=n)
    torch.testing.assert_close(out, out2)
    out3 = torch.empty_like(out)
    _randgen_kernel[(1, )](43, out3, s0o, s1o, N=n)
    assert not torch.equal(out, out3)


def test_randgen_invalid_n_out():
    out = torch.empty(8, device="txda", dtype=torch.int64)
    s0o = torch.zeros(16, device="txda", dtype=torch.int64)
    s1o = torch.zeros(16, device="txda", dtype=torch.int64)
    with pytest.raises(Exception):
        _randgen_kernel[(1, )](42, out, s0o, s1o, N=8)


def test_rand_uniform_stats():
    n = 16384
    out = torch.empty(n, device="txda", dtype=torch.float32)
    _rand_kernel[(1, )](7, out, N=n)
    assert out.min().item() >= 0.0 and out.max().item() < 1.0
    mean, std = out.mean().item(), out.std().item()
    assert abs(mean - 0.5) < 0.03, f"uniform mean off: {mean}"
    assert abs(std - 0.2887) < 0.01, f"uniform std off: {std}"


def test_randn_normal_stats():
    n = 16384
    out = torch.empty(n, device="txda", dtype=torch.float32)
    _randn_kernel[(1, )](11, out, N=n)
    mean, std = out.mean().item(), out.std().item()
    assert abs(mean) < 0.05, f"normal mean off: {mean}"
    assert abs(std - 1.0) < 0.05, f"normal std off: {std}"
    assert out.abs().max().item() > 3.0


def test_rand_invalid_n_out():
    out = torch.empty(16, device="txda", dtype=torch.float32)
    with pytest.raises(Exception):
        _rand_kernel[(1, )](7, out, N=16)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
