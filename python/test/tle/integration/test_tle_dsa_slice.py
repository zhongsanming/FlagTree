import pytest
torch = pytest.importorskip("torch")

import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from .._dsa_backend import get_dsa_backend

BACKEND, DEVICE, SPACE = get_dsa_backend()


@triton.jit
def dsa_slice_kernel(x_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr, OFF_M: tl.constexpr,
                     OFF_N: tl.constexpr, SM: tl.constexpr, SN: tl.constexpr, SPACE: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    idx = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + idx, mask=mask)

    # extract -> insert roundtrip on a zero buffer
    z = tl.zeros((BM, BN), dtype=tl.float32)
    sl = tle.dsa.extract_slice(x, (OFF_M, OFF_N), (SM, SN), (1, 1))
    out = tle.dsa.insert_slice(z, sl, (OFF_M, OFF_N), (SM, SN), (1, 1))
    tl.store(out_ptr + idx, out, mask=mask)


class TestTLEDsaSlice:

    @pytest.mark.parametrize("shape,block,off,sz", [((32, 32), (32, 32), (4, 8), (8, 8))])
    def test_extract_insert(self, shape, block, off, sz):
        torch.manual_seed(0)
        m, n = shape
        bm, bn = block
        a = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
        out = torch.zeros_like(a)
        dsa_slice_kernel[(1, )](a, out, m, n, BM=bm, BN=bn, OFF_M=off[0], OFF_N=off[1], SM=sz[0], SN=sz[1],
                                num_ctas=1, SPACE=SPACE)
        expected = torch.zeros_like(a)
        expected[off[0]:off[0] + sz[0], off[1]:off[1] + sz[1]] = a[off[0]:off[0] + sz[0], off[1]:off[1] + sz[1]]
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
