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
def _fold_reduce_kernel(X, Y0, Y1, ROWS: tl.constexpr, ROWS2: tl.constexpr,
                        HALF: tl.constexpr, FULL: tl.constexpr,
                        ROW_BLK: tl.constexpr):
    """One op, two destinations: rows [0, ROWS) -> Y0, [ROWS, ROWS2) -> Y1.

    The source is a *computed* tensor (a plain tl.load), which probe_concat
    proved raw() accepts, so nothing is staged through a buffer.  Two
    destinations rather than one vector because the caller consumes the two
    results separately and triton cannot slice a tensor.
    """
    idx = tl.arange(0, ROWS2)[:, None] * FULL + tl.arange(0, FULL)[None, :]
    src = tl.load(X + idx)

    d0 = tl.zeros([ROWS], dtype=tl.float32)
    d1 = tl.zeros([ROWS], dtype=tl.float32)
    tmp = tl.zeros([ROWS2 * HALF], dtype=tl.float32)
    d0, d1, tmp = tle.dsa.ascend.raw(
        "fold_reduce_rows", src, ROWS, ROWS, HALF, ROW_BLK,
        out=[d0, d1, tmp])
    tl.store(Y0 + tl.arange(0, ROWS), d0)
    tl.store(Y1 + tl.arange(0, ROWS), d1)


def _env_report(expect_params):
    """Print which triton/registry/bitcode is actually loaded, and whether the
    op's Python signature matches what this file calls it with.

    Added after a round was lost to "TypeError: got multiple values for argument
    'out'": the interpreter had imported an older registry.py (the runtime here
    can come from site-packages rather than the checkout), so the call was
    correct and the module was stale.  One line of self-report makes that
    obvious instead of looking like a call-site bug.
    """
    import inspect
    import os
    try:
        import triton
        import triton.experimental.tle.language.dsa.ascend.custom_ops as ops
        from triton.experimental.tle.language.dsa.ascend.custom_ops import (
            registry,
        )
        sig = inspect.signature(registry.fold_reduce_rows.__init__)
        params = [p for p in sig.parameters if p != "self"]
        bc = registry.CUSTOM_OPS_BITCODE
        print(f"  triton   : {os.path.dirname(os.path.dirname(triton.__file__))}")
        print(f"  registry : {registry.__file__}")
        print(f"  bitcode  : {bc} {'exists' if os.path.exists(bc) else 'MISSING'}"
              f" {os.path.getmtime(bc):.0f}" if os.path.exists(bc) else f"  bitcode  : {bc} MISSING")
        print(f"  op params: {params}")
        if params != expect_params:
            print(f"  MISMATCH : this file calls it with {expect_params}")
            print("             -> the imported registry.py is stale or from another")
            print("                install. Check that `import triton` resolves to the")
            print("                checkout you edited, or reinstall it.")
            return False
    except Exception as exc:  # a missing op is a result too
        print(f"  registry : ERROR {type(exc).__name__}: {exc}")
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--rows", default="1,2,16",
                   help="rows per destination; comma separated")
    p.add_argument("--mask", type=int, default=64)
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    if not _HAS_NPU:
        raise SystemExit("this needs an NPU (torch_npu)")
    torch.npu.set_device(args.device)

    half = args.mask
    full = 2 * half
    row_blk = full * 4 // 32              # 32 B datablocks per row of fp32
    ok = True
    if not _env_report(["src", "repeat0", "repeat1", "mask", "src_rep_stride", "out"]):
        raise SystemExit(2)
    print(f"device={args.device}  HALF={half}  FULL={full}")
    print(f"{'rows':>5} {'max |err| d0':>13} {'max |err| d1':>13} {'rel':>9}")

    for rows in (int(x) for x in str(args.rows).split(",")):
        rows2 = 2 * rows
        g = torch.Generator(device="cpu").manual_seed(rows)
        x = torch.randn(rows2, full, generator=g).to(args.device)
        y0 = torch.zeros(rows, device=args.device)
        y1 = torch.zeros(rows, device=args.device)
        try:
            _fold_reduce_kernel[(1,)](x, y0, y1, ROWS=rows, ROWS2=rows2,
                                      HALF=half, FULL=full, ROW_BLK=row_blk)
            torch.npu.synchronize()
            folded = x[:, :half] + x[:, half:]
            ref0 = folded[:rows].sum(1)
            ref1 = folded[rows:].sum(1)
            e0 = (y0 - ref0).abs().max().item()
            e1 = (y1 - ref1).abs().max().item()
            scale = max(ref0.abs().max().item(), ref1.abs().max().item())
            rel = max(e0, e1) / scale
            good = rel < args.tol
            ok = ok and good
            print(f"{rows:5d} {e0:13.2e} {e1:13.2e} {rel:9.2e} "
                  f"{'PASS' if good else 'FAIL'}")
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}".rstrip().splitlines()
            print(f"{rows:5d}  ERR {detail[0][:90]}")
            for line in detail[1:20]:
                print(f"           {line.strip()[:110]}")

    print()
    print("this is a correctness gate only (see the module docstring): an op call")
    print("per launch cannot be timed above the ~140 us launch floor, and a repeat")
    print("loop around it is either hoisted or needs a dependency that biases the")
    print("comparison. The cost is estimated in situ instead.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
