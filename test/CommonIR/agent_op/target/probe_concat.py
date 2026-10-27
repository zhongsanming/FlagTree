"""Can two computed [ROWS, K] tiles be concatenated into one UB buffer?

This is the single make-or-break step for wiring fold_reduce_rows into the KDA
kernel, so it is checked on its own before any kernel is touched.

Why it is needed: the two fused reductions both need two tiles *adjacent* in UB

    (s_q, s_k)            = fold_reduce over [ q*q ; k*k ]        (2 rows)
    (v_corr, out_part)    = fold_reduce over [ state*k ; state*q ] (2*BV rows)

because the op reads `repeat` rows from one buffer.  Plain triton cannot build
that tile: there is no concat, tl.join/tl.split do not lower on this backend
(6.6), and the products are computed values rather than loads.  TLE has
`tle.dsa.to_tensor` + `tle.dsa.insert_slice`, which is presumably how it is meant
to be done -- but "presumably" is what has cost us rounds before, so:

    stage 1  concat only:   does insert_slice place the tiles where we expect?
    stage 2  concat + raw:  does fold_reduce_rows accept that tensor directly
                            (the tutorial passes tensors to raw, not buffers)?

Stage 2 failing while stage 1 passes means the argument plumbing needs
to_buffer/to_tensor around the raw call; stage 1 failing means insert_slice is
not the tool and the whole wiring idea needs rethinking.

    python3 probe_concat.py
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
def _concat_kernel(A, B, OUT, ROWS: tl.constexpr, ROWS2: tl.constexpr,
                   K: tl.constexpr):
    """Concatenate two [ROWS, K] tiles along the row axis through UB."""
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


@triton.jit
def _concat_reduce_kernel(A, B, OUT, ROWS: tl.constexpr, ROWS2: tl.constexpr,
                          K: tl.constexpr, HALF: tl.constexpr,
                          ROW_BLK: tl.constexpr):
    """Same concatenation, then one fold_reduce_rows over all ROWS2 rows."""
    idx = tl.arange(0, ROWS)[:, None] * K + tl.arange(0, K)[None, :]
    a = tl.load(A + idx)
    b = tl.load(B + idx)

    full = tle.dsa.alloc([ROWS2, K], dtype=tl.float32,
                         mem_addr_space=tle.dsa.ascend.UB)
    t = tle.dsa.to_tensor(full)
    t = tle.dsa.insert_slice(t, a, [0, 0], [ROWS, K], [K, 1])
    t = tle.dsa.insert_slice(t, b, [ROWS, 0], [ROWS, K], [K, 1])

    dst_t = tl.zeros([ROWS2], dtype=tl.float32)
    tmp_t = tl.zeros([ROWS2 * HALF], dtype=tl.float32)
    dst_t, tmp_t = tle.dsa.ascend.raw(
        "fold_reduce_rows", t, ROWS2, HALF, ROW_BLK, out=[dst_t, tmp_t])
    tl.store(OUT + tl.arange(0, ROWS2), dst_t)


def _stage(name, fn):
    try:
        fn()
        torch.npu.synchronize()
        return True, ""
    except Exception as exc:  # a compile or device error is the result here
        detail = f"{type(exc).__name__}: {exc}".rstrip()
        lines = detail.splitlines()
        print(f"  {name}: FAIL  {lines[0][:100]}")
        for line in lines[1:24]:
            print(f"    {line}")
        return False, detail


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--dim", type=int, default=128)
    args = p.parse_args()

    if not _HAS_NPU:
        raise SystemExit("this needs an NPU (torch_npu)")
    torch.npu.set_device(args.device)

    ROWS, K = args.rows, args.dim
    ROWS2 = 2 * ROWS
    HALF = K // 2
    row_blk = K * 4 // 32          # 32 B datablocks per row of K fp32
    g = torch.Generator(device="cpu").manual_seed(7)
    a = torch.randn(ROWS, K, generator=g).to(args.device)
    b = torch.randn(ROWS, K, generator=g).to(args.device)
    cat = torch.cat([a, b], 0)

    print(f"device={args.device}  ROWS={ROWS}  K={K}  -> concat [{ROWS2}, {K}]")

    # --- stage 1: concatenation alone -------------------------------------
    out1 = torch.zeros(ROWS2, K, device=args.device)
    ok1, _ = _stage("stage 1 concat",
                    lambda: _concat_kernel[(1,)](a, b, out1, ROWS=ROWS,
                                                 ROWS2=ROWS2, K=K))
    if ok1:
        err = (out1 - cat).abs().max().item()
        print(f"  stage 1 concat: PASS  max|err| {err:.2e}"
              f"  {'(exact)' if err == 0 else '(MISMATCH)'}")
        ok1 = err == 0

    # --- stage 2: concatenation feeding the custom op ----------------------
    ok2 = False
    if ok1:
        out2 = torch.zeros(ROWS2, device=args.device)
        ok2, _ = _stage("stage 2 concat+raw",
                        lambda: _concat_reduce_kernel[(1,)](
                            a, b, out2, ROWS=ROWS, ROWS2=ROWS2, K=K,
                            HALF=HALF, ROW_BLK=row_blk))
        if ok2:
            ref = (cat[:, :HALF] + cat[:, HALF:]).sum(1)
            rel = (out2 - ref).abs().max().item() / ref.abs().max().item()
            print(f"  stage 2 concat+raw: PASS  rel {rel:.2e}")
            ok2 = rel < 1e-5

    print()
    print("verdict:")
    if ok1 and ok2:
        print("  both stages pass -> the wiring is feasible as planned: build")
        print("  [.. ; ..] with insert_slice, hand the tensor straight to raw,")
        print("  and the KDA kernel can fuse its reductions with no extra copies.")
    elif ok1:
        print("  concat works, raw rejects the tensor -> wrap it with")
        print("  tle.dsa.to_buffer(.., UB) then tle.dsa.to_tensor(..) around the")
        print("  call; still feasible, one more API step.")
    else:
        print("  insert_slice is not the tool (offsets/sizes/strides or dtype).")
        print("  Wiring would need another way to get two computed tiles")
        print("  adjacent in UB, i.e. the wiring plan is not viable as written.")
    raise SystemExit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
