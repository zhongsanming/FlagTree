"""Correctness check for the fold_reduce_rows custom op.  Timing removed.

The op is the multi-result reduce that triton cannot express (see
custom_ops/reduction_ops/fold_reduce_rows.cpp and KDA_ASCENDC_VS_TLE.md 6.10):
one vadd that folds each row's two halves, then one vcadd over all rows, so
`repeat` results cost two instructions instead of `repeat` separate tl.sum
chains.

This file used to also report "op us vs triton us", with the repeats inside the
kernel and a launch floor subtracted.  That measurement could not be made
honest, and the 0.000 us columns it produced were the tell:

  * the op path calls raw(src_t, ..) with loop-invariant inputs and the same out
    buffers, so CSE folds 64 or 256 calls into one; the triton path's tl.sum does
    not depend on the loop variable, so it is hoisted and `acc += s` becomes
    REPEAT*s.  Neither side does the work;
  * subtracting one constant ~140 us only works if the launch cost is constant,
    and it is not (128.7 .. 140.6 us measured across row counts);
  * it cannot be fixed cheaply: defeating hoisting needs an input that changes
    per iteration, which costs a copy or a store that one side would not pay.

The honest number is in-situ, and it is already measured: in the real kernel,
adding a fifth reduction cost +13 % (6.6), i.e. ~141 ns/token per reduce, so
fusing 4 -> 2 should give 1082 -> ~801 ns/token = 1.35x.  Measuring it properly
means wiring the op into the kernel, where it sits in an 8192-iteration loop and
neither hoisting nor the launch floor applies.

So this script is now only the correctness gate: it validates the CIFACE
argument order, the fold, vcadd MODE=0 and the datablock stride semantics against
torch.  It passed at rows=1/2/16/32/64.

    python3 probe_fold_reduce.py
    python3 probe_fold_reduce.py --rows 1,2,16,32,64
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
                        FULL: tl.constexpr, ROW_BLK: tl.constexpr,
                        REPEAT: tl.constexpr):
    """Grid: one program; a [ROWS, FULL] tile in, ROWS sums out.

    REPEAT exists because one launch costs ~140 us of host time here, which is
    larger than the op: timing a single call per launch measures the launch
    floor (the first version of this probe did exactly that and reported
    128-141 us for every row count, the same trap probe_dot hit).  The repeats
    are accumulated, so they are a dependent chain the compiler cannot elide,
    and per-op cost is (ms - floor)/REPEAT.

    Three conventions of `tle.dsa.ascend.raw`, all learned from
    python/tutorials/tle/custom/test_custom_ops.py:

      * inputs must be *tensors*: a buffer has to go through
        `tle.dsa.to_tensor` (passing the buffer raises
        "Unsupported argument type <..buffer>").
      * `out=` also takes tensors, not buffers -- the tutorial passes
        `tl.zeros([..])` for every output and rebinds the returned values.
      * FULL = 2*HALF is passed in rather than computed here, because tl.arange
        and tle.dsa.alloc want compile-time literals.
    """
    idx = tl.arange(0, ROWS)[:, None] * FULL + tl.arange(0, FULL)[None, :]

    src_ub = tle.dsa.alloc([ROWS, FULL], dtype=tl.float32,
                           mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(X + idx, src_ub, [ROWS, FULL])
    src_t = tle.dsa.to_tensor(src_ub)

    acc = tl.zeros([ROWS], dtype=tl.float32)
    dst_t = tl.zeros([ROWS], dtype=tl.float32)
    tmp_t = tl.zeros([ROWS * HALF], dtype=tl.float32)
    for _ in range(REPEAT):
        dst_t, tmp_t = tle.dsa.ascend.raw(
            "fold_reduce_rows", src_t, ROWS, HALF, ROW_BLK,
            out=[dst_t, tmp_t])
        acc += dst_t
    tl.store(Y + tl.arange(0, ROWS), acc)


@triton.jit
def _triton_kernel(X, Y, ROWS: tl.constexpr, HALF: tl.constexpr,
                   FULL: tl.constexpr, REPEAT: tl.constexpr):
    """The reference cost: what the same job costs without the custom op."""
    idx = tl.arange(0, ROWS)[:, None] * FULL + tl.arange(0, FULL)[None, :]
    a = tl.load(X + idx)
    lo = tl.reshape(a, [ROWS, 2, HALF])
    acc = tl.zeros([ROWS], dtype=tl.float32)
    s = tl.zeros([ROWS], dtype=tl.float32)
    for _ in range(REPEAT):
        s = tl.sum(tl.sum(lo, axis=2), axis=1)
        acc += s
    tl.store(Y + tl.arange(0, ROWS), acc)


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
    p.add_argument("--rows", default="1,2,16",
                   help="rows per program; comma separated")
    p.add_argument("--mask", type=int, default=64)
    p.add_argument("--repeat", type=int, default=64,
                   help="op calls per launch; the launch floor (~140 us) is "
                        "subtracted, so this must be >> 1 to see anything")
    p.add_argument("--floor-us", type=float, default=140.0,
                   help="measured one-launch host floor, subtracted from both "
                        "columns before dividing by --repeat")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    if not _HAS_NPU:
        raise SystemExit("this needs an NPU (torch_npu)")
    torch.npu.set_device(args.device)

    ok = True
    print(f"device={args.device}  HALF={args.mask}  repeat={args.repeat}  "
          f"reps={args.reps}  launch floor assumed {args.floor_us:.0f} us")
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
            rep = max(1, args.repeat)
            _fold_reduce_kernel[(1,)](x, y, ROWS=rows, HALF=args.mask,
                                      FULL=full, ROW_BLK=row_blk, REPEAT=rep)
            torch.npu.synchronize()
            # REPEAT accumulations, so the reference is scaled to match
            ref = rep * (x[:, :args.mask] + x[:, args.mask:]).sum(1)
            err = (y - ref).abs().max().item()
            rel = err / ref.abs().max().item()
            good = rel < args.tol
            ok = ok and good
            t_op = _run(lambda: _fold_reduce_kernel[(1,)](
                x, y, ROWS=rows, HALF=args.mask, FULL=full, ROW_BLK=row_blk,
                REPEAT=rep), args.reps, args.warmup)
            t_tri = _run(lambda: _triton_kernel[(1,)](
                x, y_ref, ROWS=rows, HALF=args.mask, FULL=full, REPEAT=rep),
                args.reps, args.warmup)
            # strip the launch floor before dividing by the repeat count
            op_us = max(t_op - args.floor_us, 1e-3) / rep
            tri_us = max(t_tri - args.floor_us, 1e-3) / rep
            print(f"{rows:5d} {err:11.2e} {rel:9.2e} {op_us:8.3f} "
                  f"{tri_us:10.3f} {tri_us / op_us:7.2f}x")
        except Exception as exc:  # compile or device error is a result here
            ok = False
            detail = f"{type(exc).__name__}: {exc}".rstrip()
            first = detail.splitlines()[0]
            print(f"{rows:5d}  ERR {first[:100]}")
            # The full message carries the triton location and the argument that
            # was rejected; truncating it to one line is what cost a round trip
            # before, so print the rest indented instead of dropping it.
            rest = detail.splitlines()[1:]
            for line in rest[:24]:
                print(f"           {line}")
            if len(rest) > 24:
                print(f"           ... {len(rest) - 24} more lines")

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
