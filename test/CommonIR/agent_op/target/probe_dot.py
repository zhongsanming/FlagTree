"""Probe the cube path before committing to a chunked (matmul-based) KDA.

The serial recurrence is reduce-bound: 4 cross-lane reductions per token at
~13 % per reduction (see KDA_ASCENDC_VS_TLE.md 6.5/6.6).  The only way past that
is to restate the recurrence in the chunked delta-rule form, which replaces the
per-token reductions with matmuls on the cube unit.  This probe answers, for the
exact shapes that form would need:

  1. does tl.dot compile at all on this backend, per dtype?
  2. is an fp32 accumulator available from bf16/fp16 inputs?
  3. what does it cost (TFLOP/s and per-tile latency)?

Those three answers decide the chunk shape, the precision of the triangular
solve, and whether the rewrite is viable at all.  Everything is reported, and a
failing dtype is printed rather than raised, so one run answers all of it.

    python3 probe_dot.py
    python3 probe_dot.py --shapes 64,128,256 --reps 30
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    print("torch_npu is not importable; this probe needs an NPU", file=sys.stderr)
    raise

import triton
import triton.language as tl

# Input dtype -> accumulator dtype.  The chunked algorithm wants a wide
# accumulator even from narrow inputs, so bf16->fp32 is the interesting case.
CASES = (
    ("bf16->fp32", torch.bfloat16, tl.float32),
    ("fp16->fp32", torch.float16, tl.float32),
    ("fp32->fp32", torch.float32, tl.float32),
)


@triton.jit
def _dot_kernel(A, B, C, M, N, K,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                ACC: tl.constexpr):
    """One program per (BM x BN) output tile; a single K tile, no loop."""
    pid = tl.program_id(0)
    n_tiles_n = N // BN
    pm = pid // n_tiles_n
    pn = pid % n_tiles_n

    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    a = tl.load(A + rm[:, None] * K + rk[None, :])
    b = tl.load(B + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b, out_dtype=ACC)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


def _bench(fn, warmup: int, reps: int):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return statistics.median(times)


def probe(mnk: int, block: int, reps: int, warmup: int, device: str):
    M = N = K = mnk
    rows = []
    for name, torch_dtype, acc in CASES:
        tag = f"{name:>11s} {mnk:4d}^3 b{block:<3d}"
        try:
            a = torch.randn(M, K, device=device, dtype=torch.float32).to(torch_dtype)
            b = torch.randn(K, N, device=device, dtype=torch.float32).to(torch_dtype)
            c = torch.empty(M, N, device=device, dtype=torch.float32)
            grid = ((M // block) * (N // block),)
            run = lambda: _dot_kernel[grid](  # noqa: E731
                a, b, c, M, N, K, BM=block, BN=block, BK=K, ACC=acc
            )
            ms = _bench(run, warmup, reps)
            torch.npu.synchronize()

            ref = (a.float() @ b.float())
            err = (c - ref).abs().max().item()
            rel = err / ref.abs().max().item()
            tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
            rows.append((tag, "ok", ms, tflops, rel, ""))
        except Exception as exc:  # compile failure is a result, not a crash
            rows.append((tag, "ERR", float("nan"), float("nan"), float("nan"),
                         f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--shapes", default="64,128,256",
                        help="M=N=K sizes to try, comma separated")
    parser.add_argument("--block", type=int, default=128,
                        help="BM=BN tile (BK is the whole K)")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    print(f"device={args.device}  block={args.block}  reps={args.reps}")
    print(f"{'case':>26}  {'stat':>4} {'ms':>9} {'TFLOP/s':>8} {'rel.err':>9}  note")
    for mnk in (int(x) for x in args.shapes.split(",")):
        for tag, stat, ms, tflops, rel, note in probe(
            mnk, args.block, args.reps, args.warmup, args.device
        ):
            if stat == "ok":
                print(f"{tag:>26}  {stat:>4} {ms:9.4f} {tflops:8.2f} {rel:9.2e}")
            else:
                print(f"{tag:>26}  {stat:>4} {'-':>9} {'-':>8} {'-':>9}  {note}")

    print()
    print("what to read:")
    print("  bf16->fp32 ok  : the chunked form can use the cube with a wide accumulator")
    print("  fp32->fp32 ERR : the triangular solve has to run on the vector unit, or")
    print("                   be restructured to stay in bf16 (accuracy risk)")
    print("  TFLOP/s        : if the 64^3/128^3 rows are far below the 256^3 row, the")
    print("                   chunk is too small to amortise launch/tile overhead and")
    print("                   the chunk size has to grow before the rewrite pays off")


if __name__ == "__main__":
    main()
