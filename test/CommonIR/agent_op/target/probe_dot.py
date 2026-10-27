"""Probe the cube path before committing to a chunked (matmul-based) KDA.

The serial recurrence is reduce-bound: 4 cross-lane reductions per token, each
worth ~13 % of the wall clock (see KDA_ASCENDC_VS_TLE.md 6.5/6.6).  The only way
past that is to restate the recurrence in the chunked delta-rule form, which
replaces the per-token reductions with matmuls on the cube unit.  This probe
answers, for the shapes that form would need:

  1. does tl.dot compile and run on this backend, per dtype?
  2. is an fp32 accumulator available from bf16/fp16 inputs?
  3. what does it cost (TFLOP/s, and per-tile latency at small tiles)?

Design notes, both learned the hard way:

* every case runs in its own subprocess.  An aicore abort wedges the NPU for
  the rest of the process -- the first run of this probe reported a real device
  error for one case and then "ZeroDivisionError" for the other eight, because
  every later tensor came back zero.  Isolating the cases costs a process
  start each and makes one bad tile cost one row instead of the whole run.
* the tile is clamped to the matrix and grid sizes are asserted.  A tile
  larger than the shape silently yields a zero-block launch, which on this
  device is itself a hard error -- that is what poisoned the first run.

    python3 probe_dot.py                        # all dtypes, 64/128/256
    python3 probe_dot.py --shapes 128 --dtypes bf16,fp16
    python3 probe_dot.py --shapes 64,128,256,512 --kblock 128
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time

CASES = (("bf16", "torch.bfloat16", "tl.float32"),
         ("fp16", "torch.float16", "tl.float32"),
         ("fp32", "torch.float32", "tl.float32"))

_SRC = '''
import statistics, time
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _dot_kernel(A, B, C, M, N, K,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                ACC: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles_n = N // BN
    pm = pid // n_tiles_n
    pn = pid % n_tiles_n

    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(A + rm[:, None] * K + (k0 + rk)[None, :])
        b = tl.load(B + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc, out_dtype=ACC)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="npu:0")
    p.add_argument("--mnk", type=int, required=True)
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--kblock", type=int, required=True)
    p.add_argument("--reps", type=int, required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--torch-dtype", required=True)
    p.add_argument("--acc", required=True)
    args = p.parse_args()

    torch.npu.set_device(args.device)
    torch_dtype = getattr(torch, args.torch_dtype.split(".")[-1])
    acc = getattr(tl, args.acc.split(".")[-1])

    M = N = K = args.mnk
    block = min(args.block, M, N)
    kblock = min(args.kblock, K)
    if M % block or N % block:
        print(f"RESULT bad-tile {block} does not divide {M}x{N}")
        return
    if K % kblock:
        print(f"RESULT bad-tile {kblock} does not divide K={K}")
        return
    a = torch.randn(M, K, device=args.device, dtype=torch.float32).to(torch_dtype)
    b = torch.randn(K, N, device=args.device, dtype=torch.float32).to(torch_dtype)
    c = torch.empty(M, N, device=args.device, dtype=torch.float32)
    grid = ((M // block) * (N // block),)
    assert grid[0] > 0, grid  # a zero-block launch hard-aborts the device
    run = lambda: _dot_kernel[grid](a, b, c, M, N, K, BM=block, BN=block,
                                    BK=kblock, ACC=acc)

    for _ in range(args.warmup):
        run()
    torch.npu.synchronize()
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        run()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ms = statistics.median(ts)

    ref = a.float() @ b.float()
    err = (c - ref).abs().max().item()
    denom = ref.abs().max().item()
    rel = err / denom if denom > 0 else float("inf")
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"RESULT ok {ms:.4f} {tflops:.2f} {rel:.3e} {block} {kblock}")


main()
'''


def run_case(name: str, torch_dtype: str, acc: str, mnk: int, args) -> tuple:
    """Run one (dtype, shape) in a child process; return a result tuple."""
    cmd = [
        sys.executable, "-c", _SRC,
        "--device", args.device,
        "--mnk", str(mnk),
        "--block", str(args.block),
        "--kblock", str(args.kblock),
        "--reps", str(args.reps),
        "--warmup", str(args.warmup),
        "--torch-dtype", torch_dtype,
        "--acc", acc,
    ]
    env = dict(os.environ)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        return (f"{name}->fp32 {mnk:4d}^3", "TIMEOUT", None, None, None, "")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            parts = line.split()
            if parts[1] == "ok":
                return (f"{name}->fp32 {mnk:4d}^3", "ok",
                        float(parts[2]), float(parts[3]), float(parts[4]),
                        f"b{parts[5]} k{parts[6]}")
            return (f"{name}->fp32 {mnk:4d}^3", "bad", None, None, None,
                    " ".join(parts[2:]))
    note = ""
    for line in reversed((proc.stderr or "").splitlines()):
        if line.strip():
            note = line.strip()[:110]
            break
    return (f"{name}->fp32 {mnk:4d}^3",
            f"ERR({proc.returncode})", None, None, None, note)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--shapes", default="64,128,256")
    parser.add_argument("--dtypes", default="bf16,fp16,fp32")
    parser.add_argument("--block", type=int, default=128,
                        help="BM=BN tile, clamped to the shape")
    parser.add_argument("--kblock", type=int, default=128,
                        help="BK; K is walked in steps of this")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    wanted = {d.strip() for d in args.dtypes.split(",") if d.strip()}
    shapes = [int(x) for x in args.shapes.split(",")]
    cases = [c for c in CASES if c[0] in wanted]

    print(f"device={args.device}  block={args.block}  kblock={args.kblock}  "
          f"one subprocess per case")
    print(f"{'case':>17}  {'stat':>10} {'ms':>9} {'TFLOP/s':>8} {'rel.err':>9}  note")
    for mnk in shapes:
        for name, torch_dtype, acc in cases:
            tag, stat, ms, tflops, rel, note = run_case(name, torch_dtype, acc, mnk, args)
            if stat == "ok":
                print(f"{tag:>17}  {stat:>10} {ms:9.4f} {tflops:8.2f} {rel:9.2e}  {note}")
            else:
                print(f"{tag:>17}  {stat:>10} {'-':>9} {'-':>8} {'-':>9}  {note}",
                      flush=True)

    print()
    print("what to read:")
    print("  bf16->fp32 ok  : the chunked form can use the cube with a wide accumulator")
    print("  fp32->fp32 ERR : the triangular solve has to stay on the vector unit, or")
    print("                   the chunk has to be restructured to stay in bf16")
    print("  TFLOP/s        : if 64^3/128^3 sit far below 256^3, the chunk is too small")
    print("                   to amortise tile overhead and chunk size has to grow")
    print("  any ERR        : the note is the last stderr line; rerun that single case")
    print("                   by hand to see the full device error")


if __name__ == "__main__":
    main()
