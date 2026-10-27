"""Decide how to feed two computed tiles to fold_reduce_rows, without concat.

The fused reductions need two results from one instruction:

    (s_q, s_k)          <- rows of  [ q*q ; k*k ]
    (v_corr, out_part)  <- rows of  [ state*k ; state*q ]

The obvious way to build such a tile is a concatenation, and TLE has
`to_tensor` + `insert_slice` for that -- but it faults on device ("The address for
the VEC instruction to read/write UB is out of bounds"), so that route is out as
written (stage B below, kept for the record).

There is a route that needs no concat at all, because both halves multiply the
*same* tile:

    row    = arange(0, 2*BV)
    x2     = load(X + (row % BV)[:, None] * K + arange(K))   # tile, duplicated
    w2     = where(row[:, None] < BV, k[None, :], q[None, :])
    prod   = x2 * w2                                         # [2*BV, K], computed

`row % BV` duplicates the tile, `where` picks the operand per half, and the result
is an ordinary computed tensor -- no insertion into a buffer, nothing to stride
wrongly. For the norms the same pattern gives [q*q ; k*k] with two rows.

So the only open question is whether a *computed* tensor can be handed to raw()
directly. The repo already does that for an input (sort_1d_pack takes
`tmp = tl.zeros([TMP_SIZE])`), but not for something this large, so:

    stage A  no-concat, computed tensor -> raw   (the route we would use)
    stage B  insert_slice                (known to fault; run with --with-b)

    python3 probe_concat.py
    python3 probe_concat.py --bv 32 --with-b
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
def _modular_kernel(X, K1, Q1, OUT0, OUT1, BV: tl.constexpr, BV2: tl.constexpr,
                    K: tl.constexpr, HALF: tl.constexpr,
                    ROW_BLK: tl.constexpr):
    """[x*k ; x*q] built by duplicating x, then one fold_reduce_rows.

    This is the shape the KDA kernel needs: the two results are `v_corr` and
    `out_part`, and they come back in two separate tensors so nothing has to be
    sliced afterwards.
    """
    row = tl.arange(0, BV2)
    cols = tl.arange(0, K)

    x2 = tl.load(X + (row % BV)[:, None] * K + cols[None, :])
    k1 = tl.load(K1 + cols)
    q1 = tl.load(Q1 + cols)
    w2 = tl.where(row[:, None] < BV, k1[None, :], q1[None, :])
    prod = x2 * w2                                   # [BV2, K], computed

    d0 = tl.zeros([BV], dtype=tl.float32)
    d1 = tl.zeros([BV], dtype=tl.float32)
    tmp = tl.zeros([BV2 * HALF], dtype=tl.float32)
    d0, d1, tmp = tle.dsa.ascend.raw(
        "fold_reduce_rows", prod, BV, BV, HALF, ROW_BLK, out=[d0, d1, tmp])
    tl.store(OUT0 + tl.arange(0, BV), d0)
    tl.store(OUT1 + tl.arange(0, BV), d1)


@triton.jit
def _insert_slice_kernel(A, B, OUT, ROWS: tl.constexpr, ROWS2: tl.constexpr,
                         K: tl.constexpr):
    """The concatenation route.  Faults on device; kept to document that."""
    idx = tl.arange(0, ROWS)[:, None] * K + tl.arange(0, K)[None, :]
    a = tl.load(A + idx)
    b = tl.load(B + idx)

    full = tle.dsa.alloc([ROWS2, K], dtype=tl.float32,
                         mem_addr_space=tle.dsa.ascend.UB)
    t = tle.dsa.to_tensor(full)
    t = tle.dsa.insert_slice(t, a, [0, 0], [ROWS, K], [K, 1])
    t = tle.dsa.insert_slice(t, b, [ROWS, 0], [ROWS, K], [K, 1])

    oidx = tl.arange(0, ROWS2)[:, None] * K + tl.arange(0, K)[None, :]
    tl.store(OUT + oidx, t)


def _report(name, fn, check):
    try:
        fn()
        torch.npu.synchronize()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}".rstrip()
        lines = detail.splitlines()
        print(f"  {name}: FAIL  {lines[0][:100]}")
        for line in lines[1:16]:
            print(f"    {line.strip()[:110]}")
        return False
    rel = check()
    print(f"  {name}: PASS  rel {rel:.2e}")
    return rel < 1e-5


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--bv", type=int, default=16)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--with-b", action="store_true",
                   help="also run the insert_slice route (known to fault)")
    args = p.parse_args()

    if not _HAS_NPU:
        raise SystemExit("this needs an NPU (torch_npu)")
    torch.npu.set_device(args.device)

    BV, K = args.bv, args.dim
    BV2, HALF = 2 * BV, K // 2
    row_blk = K * 4 // 32
    g = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randn(BV, K, generator=g).to(args.device)
    k1 = torch.randn(K, generator=g).to(args.device)
    q1 = torch.randn(K, generator=g).to(args.device)
    out0 = torch.zeros(BV, device=args.device)
    out1 = torch.zeros(BV, device=args.device)

    print(f"device={args.device}  BV={BV}  K={K}  HALF={HALF}")

    ref0 = (x * k1).sum(1)
    ref1 = (x * q1).sum(1)
    scale = max(ref0.abs().max().item(), ref1.abs().max().item())

    def check_a():
        e = max((out0 - ref0).abs().max().item(),
                (out1 - ref1).abs().max().item())
        return e / scale

    ok_a = _report(
        "stage A no-concat -> raw",
        lambda: _modular_kernel[(1,)](x, k1, q1, out0, out1, BV=BV, BV2=BV2,
                                      K=K, HALF=HALF, ROW_BLK=row_blk),
        check_a)

    ok_b = None
    if args.with_b:
        a = torch.randn(BV, K, generator=g).to(args.device)
        b = torch.randn(BV, K, generator=g).to(args.device)
        out_b = torch.zeros(BV2, K, device=args.device)
        cat = torch.cat([a, b], 0)
        ok_b = _report(
            "stage B insert_slice concat",
            lambda: _insert_slice_kernel[(1,)](a, b, out_b, ROWS=BV,
                                               ROWS2=BV2, K=K),
            lambda: (out_b - cat).abs().max().item() / cat.abs().max().item())

    print()
    if ok_a:
        print("verdict: a computed tensor can go straight into raw, so the fused")
        print("  reductions are wireable with NO buffer construction at all:")
        print("  duplicate the tile with `row % BV`, pick the operand with `where`,")
        print("  reduce all 2*BV rows in one call. Expected ~1.35x in the kernel.")
    else:
        print("verdict: neither route feeds two tiles to the op. Wiring the fused")
        print("  reduce would need a buffer-write mechanism we do not have, so")
        print("  this line of work stops here: ship v1 (1.04-1.065x over v0) and")
        print("  keep the analysis. The reduce-cost model still stands.")
    raise SystemExit(0 if ok_a else 1)


if __name__ == "__main__":
    main()
