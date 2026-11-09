# flagtree tle
"""TLE DSA pipeline end-to-end integration (backend-portable: txda/npu)."""

import pytest
torch = pytest.importorskip("torch")

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from .._dsa_backend import get_dsa_backend

BACKEND, DEVICE, SPACE = get_dsa_backend()


@triton.jit
def elementwise_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    xnumel,
    ynumel,
    xstride_a,
    ystride_a,
    xstride_b,
    ystride_b,
    xstride_c,
    ystride_c,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    xoffs = pid * XBLOCK + tl.arange(0, XBLOCK)
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    a_smem = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32)
    b_smem = tle.dsa.alloc([XBLOCK, YBLOCK], dtype=tl.float32)
    row_ids = tl.arange(0, XBLOCK)[:, None]
    col_ids = tl.arange(0, YBLOCK)[None, :]
    row_ids = tl.broadcast_to(row_ids, (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(col_ids, (XBLOCK, YBLOCK))
    a_smem_ptrs = tle.dsa.local_ptr(a_smem, (row_ids, col_ids))
    b_smem_ptrs = tle.dsa.local_ptr(b_smem, (row_ids, col_ids))

    for yoff in tle.dsa.pipeline(0, ynumel, YBLOCK, num_stages=2):
        yoffs = tl.arange(0, YBLOCK) + yoff
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]
        tle.dsa.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
        tle.dsa.copy(b_ptrs + ystride_b * yoffs[None, :], b_smem, [XBLOCK, YBLOCK])
        aval = tl.load(a_smem_ptrs)
        bval = tl.load(b_smem_ptrs)
        c_val = aval + bval
        tl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add(A, B, C, XBLOCK=32, YBLOCK=64):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_add_kernel[grid](A, B, C, xnumel, ynumel, *A.stride(), *B.stride(), *C.stride(), XBLOCK, YBLOCK)


class TestTLEPipelineEndToEnd:

    def test_elementwise_add_basic(self):
        torch.manual_seed(42)
        xnumel, ynumel = 512, 512
        XBLOCK, YBLOCK = 64, 64
        a = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
        b = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
        c = torch.empty_like(a)
        elementwise_add(a, b, c, XBLOCK, YBLOCK)
        torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)

    def test_elementwise_add_different_sizes(self):
        torch.manual_seed(123)
        for xnumel, ynumel, XBLOCK, YBLOCK in [(256, 256, 32, 32), (1024, 1024, 64, 64), (512, 2048, 32, 128)]:
            a = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
            b = torch.randn(xnumel, ynumel, device=DEVICE, dtype=torch.float32)
            c = torch.empty_like(a)
            elementwise_add(a, b, c, XBLOCK, YBLOCK)
            torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)

    def test_elementwise_add_edge_cases(self):
        torch.manual_seed(789)
        a = torch.randn(1, 1, device=DEVICE, dtype=torch.float32)
        b = torch.randn(1, 1, device=DEVICE, dtype=torch.float32)
        c = torch.empty_like(a)
        elementwise_add(a, b, c, 1, 1)
        torch.testing.assert_close(c, a + b, atol=1e-5, rtol=1e-5)

    def test_tle_module_import(self):
        assert hasattr(tle, "dsa")
        assert hasattr(tle.dsa, "alloc")
        assert hasattr(tle.dsa, "copy")
        assert hasattr(tle.dsa, "local_ptr")
        assert hasattr(tle.dsa, "pipeline")
        assert hasattr(tle.dsa, "buffered_tensor")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
