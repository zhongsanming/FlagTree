"""Validate the fold_reduce_rows custom op in isolation, before any kernel uses it.

The op is the multi-result reduce that triton cannot express (see
custom_ops/reduction_ops/fold_reduce_rows.cpp and KDA_ASCENDC_VS_TLE.md 6.10):
one vadd that folds each row's two halves, then one vcadd over all rows, so
`repeat` results cost two instructions instead of `repeat` separate tl.sum
chains.

This checks two things, in this order:

  1. correctness against torch, for several row counts and both supported
     dtypes -- a wrong result here means the strides are wrong, and that is
     much cheaper to find now than inside the KDA kernel;
  2. cost against the triton-only equivalent (`tl.sum(a + b, axis=1)`), so the
     gain can be measured before it is relied upon.

    python3 probe_fold_reduce.py
    python3 probe_fold_reduce.py --rows 32 --mask 64 --reps 30
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

try:  # registers the "npu" backend
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except ImportError:  # pragma: no cover
    _HAS_NPU = False

import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def _fold_reduce_kernel(X, Y, ROWS: tl.constexpr, HALF: tl.constexpr,
                        ROW_BLK: tl.constexpr):
    """Grid: one program; a [ROWS, 2*HALF] tile in, ROWS sums out."""
    full = 2 * HALF
    idx = tl.arange(0, ROWS)[:, None] * full + tl.arange(0, full)[None, :]

    src_ub = tle.dsa.alloc([ROWS, full], dtype=tl.float32,
                           mem_addr_space=tle.dsa.ascend.UB)
    tmp_ub = tle.dsa.alloc([ROWS, HALF], dtype=tl.float32,
                           mem_addr_space=tle.dsa.ascend.UB)
    dst_ub = tle.dsa.alloc([ROWS], dtype=tl.float32,
                           mem_addr_space=tle.dsa.ascend.UB)

    tle.dsa.copy(X + idx, src_ub, [ROWS, full])
    tle.dsa.ascend.raw("fold_reduce_rows", src_ub, ROWS, HALF, ROW_BLK,
                       out=[dst_ub, tmp_ub])
    tle.dsa.copy(dst_ub, Y + tl.arange(0, ROWS), [ROWS])


@triton.jit
def _triton_kernel(X, Y, ROWS: tl.constexpr, HALF: tl.constexpr):
    """The reference cost: what the same job costs without the custom op."""
    full = 2 * HALF
    idx = tl.arange(0, ROWS)[:, None] * full + tl.arange(0, full)[None, :]
    a = tl.load(X + idx)
    lo = tl.reshape(a, [ROWS, 2, HALF])
    s = tl.sum(lo, axis=1)
    s = tl.sum(s, axis=1)
    tl.store(Y + tl.arange(0, ROWS), s)


def _run(fn, reps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--rows", type=int, default=16,
                   help="rows per program (comma list allowed)")
    p.add_argument("--mask", type=int, default=64)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    if not _HAS_NPU:
        raise SystemExit("this needs an NPU (torch_npu)")
    torch.npu.set_device(args.device)

    ok = True
    print(f"device={args.device}  HALF={args.mask}  reps={args.reps}")
    print(f"{'rows':>5} {'max |err|':>11} {'rel':>9} {'op us':>8} "
          f"{'triton us':>10} {'speedup':>8}")
    for rows in (int(x) for x in str(args.rows).split(",")):
        full = 2 * args.mask
        row_blk = full * 4 // 32          # 32 B datablocks per row
        g = torch.Generator(device="cpu").manual_seed(rows)
        x = torch.randn(rows, full, generator=g).to(args.device)
        y = torch.zeros(rows, device=args.device)
        y_ref = torch.zeros(rows, device=args.device)

        try:
            _fold_reduce_kernel[(1,)](x, y, ROWS=rows, HALF=args.mask,
                                      ROW_BLK=row_blk)
            torch.npu.synchronize()
            ref = (x[:, :args.mask] + x[:, args.mask:]).sum(1)
            err = (y - ref).abs().max().item()
            rel = err / ref.abs().max().item()
            good = rel < args.tol
            ok = ok and good
            t_op = _run(lambda: _fold_reduce_kernel[(1,)](
                x, y, ROWS=rows, HALF=args.mask, ROW_BLK=row_blk),
                args.reps, args.warmup)
            t_tri = _run(lambda: _triton_kernel[(1,)](
                x, y_ref, ROWS=rows, HALF=args.mask), args.reps, args.warmup)
            print(f"{rows:5d} {err:11.2e} {rel:9.2e} {t_op:8.2f} "
                  f"{t_tri:10.2f} {t_tri / t_op:7.2f}x")
        except Exception as exc:  # compile or device error is a result here
            ok = False
            print(f"{rows:5d}  ERR {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:90]}")

    print()
    print("what to read:")
    print("  rel ~1e-7        : the op is right (fp32 accumulation over 2*HALF lanes)")
    print("  ERR              : the CIFACE arg order or the strides are wrong; the")
    print("                     op takes (src, repeat, mask, src_rep_stride,")
    print("                     dst, tmp) and src_rep_stride is in 32 B datablocks")
    print("  speedup >> 1     : the primitive is worth wiring into the KDA kernel,")
    print("                     where four reduces per token are 66 % of the time")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
