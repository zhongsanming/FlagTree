# flagtree tle
"""TLE DSA slice family tests, portable across tsingmicro (txda) and Ascend NPU.

Covers `tle.dsa.extract_slice`/`insert_slice` (element-level strided slicing)
and the generic tle-lite `tle.extract_tile`/`tle.insert_tile` grid-coordinate
forms, which lower through the shared tle dialect and resolve to the same
dsa.extract_slice / dsa.insert_slice IR ops on this backend.
"""

import pytest

torch = pytest.importorskip("torch")

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from .._dsa_backend import get_dsa_backend

BACKEND, DEVICE, SPACE = get_dsa_backend()
IS_ASCEND = BACKEND in ("npu", "ascend")


@triton.jit
def _idx(shape: tl.constexpr):
    return (tl.arange(0, shape[0])[:, None, None, None] * shape[1] * shape[2] * shape[3] +
            tl.arange(0, shape[1])[None, :, None, None] * shape[2] * shape[3] +
            tl.arange(0, shape[2])[None, None, :, None] * shape[3] + tl.arange(0, shape[3])[None, None, None, :])


TILE_SRC = (32, 32, 32, 16)  # tile-test source shape
TILE = (16, 16, 16, 8)  # tile shape -> grid [2, 2, 2, 2]
SLICE = (16, 16, 16, 16)  # slice-test source shape

# -------------------------------- tile kernels -------------------------------


@triton.jit
def extract_scalar(x_ptr, out_ptr, shape: tl.constexpr, tile_shape: tl.constexpr, LIN: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    tile = tle.extract_tile(x, index=LIN, tile_shape=tile_shape)
    tl.store(out_ptr + _idx(tile_shape), tile)


@triton.jit
def extract_dyn_multi(x_ptr, out_ptr, shape: tl.constexpr, tile_shape: tl.constexpr, I0: tl.constexpr, I1: tl.constexpr,
                      I2: tl.constexpr, I3: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    r = tl.full((), I0, tl.int32)
    c = tl.full((), I1, tl.int32)
    tile = tle.extract_tile(x, index=[r, c, I2, I3], tile_shape=tile_shape)
    tl.store(out_ptr + _idx(tile_shape), tile)


@triton.jit
def insert_multi(x_ptr, out_ptr, shape: tl.constexpr, tile_shape: tl.constexpr, SI0: tl.constexpr, SI1: tl.constexpr,
                 SI2: tl.constexpr, SI3: tl.constexpr, DI0: tl.constexpr, DI1: tl.constexpr, DI2: tl.constexpr,
                 DI3: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    tile = tle.extract_tile(x, index=[SI0, SI1, SI2, SI3], tile_shape=tile_shape)
    tile = tile + 1.0
    y = tle.insert_tile(x, tile, index=[DI0, DI1, DI2, DI3])
    tl.store(out_ptr + _idx(shape), y)


@triton.jit
def insert_oop(x_ptr, out_ptr, shape: tl.constexpr, tile_shape: tl.constexpr, DI0: tl.constexpr, DI1: tl.constexpr,
               DI2: tl.constexpr, DI3: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    tile = tle.extract_tile(x, index=[0, 0, 0, 0], tile_shape=tile_shape)
    t2 = tile + 1.0
    s = tl.sum(tile)  # forces out-of-place mk.addvs: original tile still needed
    y = tle.insert_tile(x, t2, index=[DI0, DI1, DI2, DI3])
    y = y + s
    tl.store(out_ptr + _idx(shape), y)


@triton.jit
def insert_dyn_scalar(x_ptr, idx_src_ptr, idx_dst_ptr, out_ptr, shape: tl.constexpr, tile_shape: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    idx_src = tl.load(idx_src_ptr)
    idx_dst = tl.load(idx_dst_ptr)
    tile = tle.extract_tile(x, index=idx_src, tile_shape=tile_shape)
    tile = tile + 1.0
    y = tle.insert_tile(x, tile, index=idx_dst)
    tl.store(out_ptr + _idx(shape), y)


# -------------------------------- slice kernels ------------------------------


@triton.jit
def extract_static(x_ptr, out_ptr, shape: tl.constexpr, offsets: tl.constexpr, sizes: tl.constexpr,
                   strides: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    sub = tle.dsa.extract_slice(x, offsets=offsets, sizes=sizes, strides=strides)
    tl.store(out_ptr + _idx(sizes), sub)


@triton.jit
def extract_dyn(x_ptr, o0_ptr, o1_ptr, out_ptr, shape: tl.constexpr, sizes: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    o0 = tl.load(o0_ptr)
    o1 = tl.load(o1_ptr)
    sub = tle.dsa.extract_slice(x, offsets=(o0, o1, 0, 0), sizes=sizes, strides=(1, 1, 1, 1))
    tl.store(out_ptr + _idx(sizes), sub)


@triton.jit
def extract_mixed(x_ptr, o0_ptr, out_ptr, shape: tl.constexpr, sizes: tl.constexpr, O1: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    o0 = tl.load(o0_ptr)
    sub = tle.dsa.extract_slice(x, offsets=(o0, O1, 0, 0), sizes=sizes, strides=(1, 1, 1, 1))
    tl.store(out_ptr + _idx(sizes), sub)


@triton.jit
def insert_default(x_ptr, out_ptr, shape: tl.constexpr, offsets: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    tile = tle.dsa.extract_slice(x, offsets=(0, 0, 0, 0), sizes=(4, 4, 4, 4), strides=(1, 1, 1, 1))
    tile = tile + 1.0
    y = tle.dsa.insert_slice(x, tile, offsets=offsets)
    tl.store(out_ptr + _idx(shape), y)


@triton.jit
def insert_strided(x_ptr, out_ptr, shape: tl.constexpr, offsets: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    tile = tle.dsa.extract_slice(x, offsets=(0, 0, 0, 0), sizes=(4, 4, 4, 4), strides=(1, 1, 1, 1))
    tile = tile + 1.0
    y = tle.dsa.insert_slice(x, tile, offsets=offsets, sizes=(4, 4, 4, 4), strides=(2, 2, 2, 2))
    tl.store(out_ptr + _idx(shape), y)


@triton.jit
def member_roundtrip(x_ptr, out_ptr, shape: tl.constexpr, offsets: tl.constexpr):
    x = tl.load(x_ptr + _idx(shape))
    sub = x.extract_slice(offsets=(0, 0, 0, 0), sizes=(8, 8, 8, 8), strides=(1, 1, 1, 1))
    sub = sub + 1.0
    y = x.insert_slice(sub, offsets=offsets)
    tl.store(out_ptr + _idx(shape), y)


@pytest.mark.skipif(
    IS_ASCEND,
    reason="tle.extract_tile/insert_tile leave the TLE dialect in the ttadapter IR, "
    "but the Ascend binary pipeline does not register dialect 'tle'",
)
class TestTile:

    def test_extract_scalar(self):
        torch.manual_seed(42)
        x = torch.randn(*TILE_SRC, device=DEVICE, dtype=torch.float32)

        # LIN=15 == linear id of [1,1,1,1]
        out = torch.zeros(*TILE, device=DEVICE, dtype=torch.float32)
        extract_scalar[(1, )](x, out, TILE_SRC, TILE, 15)
        torch.testing.assert_close(out, x[16:32, 16:32, 16:32, 8:16])

    def test_extract_dyn_multi(self):
        torch.manual_seed(42)
        x = torch.randn(*TILE_SRC, device=DEVICE, dtype=torch.float32)

        # mixed: dims 0,1 dynamic, dims 2,3 static
        out = torch.zeros(*TILE, device=DEVICE, dtype=torch.float32)
        extract_dyn_multi[(1, )](x, out, TILE_SRC, TILE, 1, 1, 0, 0)
        torch.testing.assert_close(out, x[16:32, 16:32, 0:16, 0:8])

    def test_insert_multi(self):
        torch.manual_seed(44)
        shape, tile = (16, 16, 16, 16), (8, 8, 8, 8)
        x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)

        # identity: extract [0,0,0,0], +1.0, insert back at [0,0,0,0]
        out = torch.zeros(*shape, device=DEVICE, dtype=torch.float32)
        insert_multi[(1, )](x, out, shape, tile, 0, 0, 0, 0, 0, 0, 0, 0)
        expected = x.clone()
        expected[0:8, 0:8, 0:8, 0:8] += 1.0
        torch.testing.assert_close(out, expected)

        # relocate: extract [1,1,1,1], +1.0, insert at [0,0,0,0]
        out = torch.zeros(*shape, device=DEVICE, dtype=torch.float32)
        insert_multi[(1, )](x, out, shape, tile, 1, 1, 1, 1, 0, 0, 0, 0)
        expected = x.clone()
        expected[0:8, 0:8, 0:8, 0:8] = x[8:16, 8:16, 8:16, 8:16] + 1.0
        torch.testing.assert_close(out, expected)

    def test_insert_oop(self):
        torch.manual_seed(47)
        shape, tile = (16, 16, 16, 16), (8, 8, 8, 8)
        x2 = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
        out = torch.zeros(*shape, device=DEVICE, dtype=torch.float32)
        insert_oop[(1, )](x2, out, shape, tile, 1, 1, 1, 1)
        src = x2[0:8, 0:8, 0:8, 0:8]
        expected = x2.clone()
        expected[8:16, 8:16, 8:16, 8:16] = src + 1.0
        expected = expected + src.sum()
        torch.testing.assert_close(out, expected)

    def test_insert_dyn_scalar(self):
        torch.manual_seed(50)
        shape, tile = (16, 16, 16, 16), (8, 8, 8, 8)

        # Runtime indices (5=[0,1,0,1], 10=[1,0,1,0]) exercise the div/mod path.
        x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
        idx_src = torch.tensor(5, device=DEVICE, dtype=torch.int32)
        idx_dst = torch.tensor(10, device=DEVICE, dtype=torch.int32)
        out = torch.zeros(*shape, device=DEVICE, dtype=torch.float32)
        insert_dyn_scalar[(1, )](x, idx_src, idx_dst, out, shape, tile)
        expected = x.clone()
        expected[8:16, 0:8, 8:16, 0:8] = x[0:8, 8:16, 0:8, 8:16] + 1.0
        torch.testing.assert_close(out, expected)


@pytest.mark.skipif(
    IS_ASCEND,
    reason="4D slice forms are not yet supported by the Ascend legacy frontend "
    "(constexpr tuple offsets / default sizes) and exceed the 950 UB budget; "
    "TestTLEDsaSlice2D covers Ascend",
)
class TestSlice:

    def test_extract_static(self):
        torch.manual_seed(42)
        x = torch.randn(*SLICE, device=DEVICE, dtype=torch.float32)

        # stride 1
        out = torch.zeros(8, 8, 8, 8, device=DEVICE, dtype=torch.float32)
        extract_static[(1, )](x, out, SLICE, (4, 4, 4, 4), (8, 8, 8, 8), (1, 1, 1, 1))
        torch.testing.assert_close(out, x[4:12, 4:12, 4:12, 4:12])

        # stride 2
        out = torch.zeros(8, 8, 8, 8, device=DEVICE, dtype=torch.float32)
        extract_static[(1, )](x, out, SLICE, (0, 0, 0, 0), (8, 8, 8, 8), (2, 2, 2, 2))
        torch.testing.assert_close(out, x[0:16:2, 0:16:2, 0:16:2, 0:16:2])

    def test_extract_dyn(self):
        torch.manual_seed(42)
        x = torch.randn(*SLICE, device=DEVICE, dtype=torch.float32)
        o0 = torch.tensor(4, device=DEVICE, dtype=torch.int32)
        o1 = torch.tensor(4, device=DEVICE, dtype=torch.int32)

        # all-dynamic offsets on dims 0,1
        out = torch.zeros(8, 8, 8, 8, device=DEVICE, dtype=torch.float32)
        extract_dyn[(1, )](x, o0, o1, out, SLICE, (8, 8, 8, 8))
        torch.testing.assert_close(out, x[4:12, 4:12, 0:8, 0:8])

        # mixed: dynamic dim0, static dim1 (=8)
        out = torch.zeros(8, 8, 8, 8, device=DEVICE, dtype=torch.float32)
        extract_mixed[(1, )](x, o0, out, SLICE, (8, 8, 8, 8), 8)
        torch.testing.assert_close(out, x[4:12, 8:16, 0:8, 0:8])

    def test_insert_default(self):
        torch.manual_seed(44)
        x = torch.randn(*SLICE, device=DEVICE, dtype=torch.float32)

        out = torch.zeros(*SLICE, device=DEVICE, dtype=torch.float32)
        insert_default[(1, )](x, out, SLICE, (8, 8, 8, 8))
        expected = x.clone()
        expected[8:12, 8:12, 8:12, 8:12] = x[0:4, 0:4, 0:4, 0:4] + 1.0
        torch.testing.assert_close(out, expected)

    def test_insert_strided(self):
        torch.manual_seed(44)
        x = torch.randn(*SLICE, device=DEVICE, dtype=torch.float32)

        out = torch.zeros(*SLICE, device=DEVICE, dtype=torch.float32)
        insert_strided[(1, )](x, out, SLICE, (4, 4, 4, 4))
        expected = x.clone()
        expected[4:12:2, 4:12:2, 4:12:2, 4:12:2] = x[0:4, 0:4, 0:4, 0:4] + 1.0
        torch.testing.assert_close(out, expected)

    def test_member(self):
        torch.manual_seed(44)
        x = torch.randn(*SLICE, device=DEVICE, dtype=torch.float32)

        out = torch.zeros(*SLICE, device=DEVICE, dtype=torch.float32)
        member_roundtrip[(1, )](x, out, SLICE, (8, 8, 8, 8))
        expected = x.clone()
        expected[8:16, 8:16, 8:16, 8:16] = x[0:8, 0:8, 0:8, 0:8] + 1.0
        torch.testing.assert_close(out, expected)


@triton.jit
def dsa_slice_2d_kernel(x_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr, OFF_M: tl.constexpr,
                        OFF_N: tl.constexpr, SM: tl.constexpr, SN: tl.constexpr):
    """2D extract -> insert round-trip on a zero block."""
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    idx = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + idx, mask=mask)

    z = tl.zeros((BM, BN), dtype=tl.float32)
    sub = tle.dsa.extract_slice(x, (OFF_M, OFF_N), (SM, SN), (1, 1))
    out = tle.dsa.insert_slice(z, sub, (OFF_M, OFF_N), (SM, SN), (1, 1))
    tl.store(out_ptr + idx, out, mask=mask)


class TestTLEDsaSlice2D:
    """Portable 2D slice test that runs on both txda and Ascend NPU."""

    @pytest.mark.parametrize("shape,block,off,sz", [((32, 32), (32, 32), (4, 8), (8, 8))])
    def test_extract_insert(self, shape, block, off, sz):
        torch.manual_seed(0)
        m, n = shape
        bm, bn = block
        a = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
        out = torch.zeros_like(a)

        launch_kwargs = {} if IS_ASCEND else {"num_ctas": 1}
        dsa_slice_2d_kernel[(1, )](a, out, m, n, BM=bm, BN=bn, OFF_M=off[0], OFF_N=off[1], SM=sz[0], SN=sz[1],
                                   **launch_kwargs)

        expected = torch.zeros_like(a)
        expected[off[0]:off[0] + sz[0], off[1]:off[1] + sz[1]] = a[off[0]:off[0] + sz[0], off[1]:off[1] + sz[1]]
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
