"""Probe the cube path before committing to a chunked (matmul-based) KDA.

The serial recurrence is reduce-bound: 4 cross-lane reductions per token, each
worth ~13 % of the wall clock (see KDA_ASCENDC_VS_TLE.md 6.5/6.6).  The only way
past that is to restate the recurrence in the chunked delta-rule form, which
replaces the per-token reductions with matmuls on the cube unit.  This probe
answers, for the shapes that form would need:

  1. does tl.dot compile and run on this backend, per dtype?
  2. is an fp32 accumulator available from bf16/fp16 inputs?
  3. what does it cost (TFLOP/s, and per-tile latency at small tiles)?

Three things were learned building this, all of them load-bearing:

* every case runs in its own subprocess, re-invoking *this file*.  An aicore
  abort wedges the NPU for the rest of the process: the first run reported one
  real device error and then "ZeroDivisionError" for the other eight, because
  every later tensor came back zero.
* the child must be this file, not `python -c`: a @triton.jit kernel needs a
  real source path, and `-c` gives `__file__ = "<string>"` ("@jit functions
  should be defined in a Python file").
* the tile is clamped and grid sizes asserted.  A tile larger than the shape
  yields a zero-block launch, which on this device is itself a hard abort.

Measurement modes.  With `--repeat 1` a launch costs ~0.14 ms of host-side
python/dispatch whatever the shape, so 64^3 and 256^3 both read 0.14 ms and the
only usable number is a lower bound on throughput.  Use:

    python3 probe_dot.py --shapes 64,128 --dtypes bf16 --repeat 256
        -> chained (dependent) dots inside one launch: per-dot latency, which
           is what a recurrence actually pays, with the host floor amortised
    python3 probe_dot.py --shapes 1024,2048 --dtypes bf16
        -> large tiles, one launch: achievable throughput

    python3 probe_dot.py                              # all dtypes, 64/128/256
    python3 probe_dot.py --shapes 128 --dtypes bf16    # one answer, seconds
    python3 probe_dot.py --shapes 64,128,256,512 --kblock 128
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
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

# name -> (input dtype, accumulator dtype).  The chunked algorithm wants a wide
# accumulator even from narrow inputs, so bf16->fp32 is the interesting case.
CASES = {
    "bf16": (torch.bfloat16, tl.float32),
    "fp16": (torch.float16, tl.float32),
    "fp32": (torch.float32, tl.float32),
}


@triton.jit
def _dot_kernel(A, B, C, M, N, K, REPEAT,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                ACC: tl.constexpr):
    """One program per (BM x BN) tile, K in BK steps, repeated REPEAT times.

    REPEAT accumulates into the same accumulator, so the dots are a *dependent*
    chain: that measures per-dot latency rather than peak throughput, and it is
    the number a recurrence actually pays.  The host-side launch floor is
    amortised over REPEAT, which is the whole point at these tile sizes.
    """
    pid = tl.program_id(0)
    n_tiles_n = N // BN
    pm = pid // n_tiles_n
    pn = pid % n_tiles_n

    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    acc = tl.zeros([BM, BN], tl.float32)
    for _ in range(REPEAT):
        for k0 in range(0, K, BK):
            a = tl.load(A + rm[:, None] * K + (k0 + rk)[None, :])
            b = tl.load(B + (k0 + rk)[:, None] * N + rn[None, :])
            acc = tl.dot(a, b, acc, out_dtype=ACC)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


# ---------------------------------------------------------------------------
# child: run exactly one case and print a single RESULT line
# ---------------------------------------------------------------------------
def run_child(args) -> int:
    torch.npu.set_device(args.device)
    torch_dtype, acc = CASES[args.dtype_tag]

    M = N = K = args.mnk
    block = min(args.block, M, N)
    kblock = min(args.kblock, K)
    if M % block or N % block or K % kblock:
        print(f"RESULT bad-tile block={block} kblock={kblock} for {M}x{N}x{K}")
        return 0

    a = torch.randn(M, K, device=args.device, dtype=torch.float32).to(torch_dtype)
    b = torch.randn(K, N, device=args.device, dtype=torch.float32).to(torch_dtype)
    c = torch.empty(M, N, device=args.device, dtype=torch.float32)

    grid = ((M // block) * (N // block),)
    assert grid[0] > 0, grid  # a zero-block launch hard-aborts the device
    rep = max(1, args.repeat)
    run = lambda: _dot_kernel[grid](  # noqa: E731
        a, b, c, M, N, K, rep, BM=block, BN=block, BK=kblock, ACC=acc
    )

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

    ref = rep * (a.float() @ b.float())
    err = (c - ref).abs().max().item()
    denom = ref.abs().max().item()
    rel = err / denom if denom > 0 else float("inf")
    tflops = 2 * M * N * K * rep / (ms * 1e-3) / 1e12
    per_dot_us = ms * 1e3 / rep
    print(f"RESULT ok {ms:.4f} {tflops:.2f} {rel:.3e} block={block} "
          f"kblock={kblock} repeat={rep} per_dot_us={per_dot_us:.3f}")
    return 0


# ---------------------------------------------------------------------------
# parent: one subprocess per case
# ---------------------------------------------------------------------------
def run_case(tag: str, mnk: int, args) -> tuple:
    cmd = [
        sys.executable, os.path.abspath(__file__), "--child",
        "--device", args.device,
        "--dtype-tag", tag,
        "--mnk", str(mnk),
        "--block", str(args.block),
        "--kblock", str(args.kblock),
        "--reps", str(args.reps),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ]
    label = f"{tag}->fp32 {mnk:4d}^3"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return (label, "TIMEOUT", None, None, None, "case exceeded 600 s")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            parts = line.split()
            if parts[1] == "ok":
                return (label, "ok", float(parts[2]), float(parts[3]),
                        float(parts[4]), " ".join(parts[5:]))
            return (label, "bad", None, None, None, " ".join(parts[2:]))
    note = ""
    for line in reversed((proc.stderr or "").splitlines()):
        if line.strip():
            note = line.strip()[:110]
            break
    status = f"CRASH({-proc.returncode})" if proc.returncode < 0 else f"ERR({proc.returncode})"
    return (label, status, None, None, None, note)


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
    parser.add_argument("--repeat", type=int, default=1,
                        help="chained dots inside one launch; amortises the "
                             "~0.14 ms host floor and exposes per-dot latency")
    # internal: the parent re-invokes this file to run one case in isolation
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dtype-tag", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mnk", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not _HAS_NPU:
        raise SystemExit("torch_npu is not importable; this probe needs an NPU")

    if args.child:
        raise SystemExit(run_child(args))

    wanted = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    unknown = [d for d in wanted if d not in CASES]
    if unknown:
        raise SystemExit(f"unknown dtype {unknown}; known: {sorted(CASES)}")
    shapes = [int(x) for x in args.shapes.split(",")]

    print(f"device={args.device}  block={args.block}  kblock={args.kblock}  "
          f"one subprocess per case")
    print(f"{'case':>17}  {'stat':>10} {'ms':>9} {'TFLOP/s':>8} {'rel.err':>9}  note")
    for mnk in shapes:
        for tag in wanted:
            label, stat, ms, tflops, rel, note = run_case(tag, mnk, args)
            if stat == "ok":
                print(f"{label:>17}  {stat:>10} {ms:9.4f} {tflops:8.2f} "
                      f"{rel:9.2e}  {note}", flush=True)
            else:
                print(f"{label:>17}  {stat:>10} {'-':>9} {'-':>8} {'-':>9}  {note}",
                      flush=True)

    print()
    print("what to read:")
    print("  bf16->fp32 ok  : the chunked form can use the cube with a wide accumulator")
    print("  fp32->fp32 ERR : the triangular solve has to stay on the vector unit, or")
    print("                   the chunk has to be restructured to stay in bf16")
    print("  per_dot_us     : with --repeat, the dependent-chain cost of one dot;")
    print("                   compare against a token's budget (~0.8 us fixed + 0.018*BV)")
    print("  TFLOP/s        : with --repeat 1 and big shapes, achievable throughput;")
    print("                   at 64^3/128^3 the host floor dominates and the number is")
    print("                   only a lower bound")
    print("  any ERR        : the note is the child's last stderr line; rerun that case")
    print("                   by hand (add --child --dtype-tag <t> --mnk <n>) for the")
    print("                   full device error")


if __name__ == "__main__":
    main()
