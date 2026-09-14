"""Benchmark mhc_pre_clamp_sinkhorn with and without the triton-lane-vectorize pass.

The ``triton-lane-vectorize`` TTIR pass (see
``lib/Dialect/Triton/Transforms/LaneVectorize.cpp``) packs lane-parallel
tensors into a leading dimension. This script measures its performance impact
on the ``mhc_pre_clamp_sinkhorn`` TLE kernel by running it twice:

    LV on   the pass is in the Ascend TTIR pipeline (default)
    LV off  ``TRITON_DISABLE_LANE_VECTORIZE=1`` skips it

Triton's JIT caches compiled kernels per process, so the toggle only takes
effect on the first compile. To avoid stale in-process caches, the two
variants are run in *separate subprocesses*, each with its own env var and a
fresh ``TRITON_CACHE_DIR``; the driver process then prints a comparison.

Timing modes (same as bench_fa_triton_arch.py)
----------------------------------------------
wall    (default) end-to-end latency: sync -> perf_counter -> kernel -> sync.
kernel  kernel-only via ``do_bench_npu`` from testing.py (mspti -> profiler).

Metrics
-------
- Latency (ms)
- Bandwidth (GB/s): read x + phi + alpha + base, write y + post_out + comb_frag.
- Speedup: latency(LV off) / latency(LV on).  > 1 means the pass helps.

Usage
-----
    # compare LV on vs off for the default shape
    python bench_mhc_pre_lane_vectorize.py

    # specific shape / sinkhorn settings / timing mode
    python bench_mhc_pre_lane_vectorize.py --B 2 --S 1024 --D 3584 \
        --iter-times 20 --clamp-min 0 --clamp-max 1 --mode kernel

    # sweep a set of shapes
    python bench_mhc_pre_lane_vectorize.py --sweep

    # run a single variant directly (no comparison, useful for debugging)
    python bench_mhc_pre_lane_vectorize.py --variant on --check
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MHC_DIR = os.path.abspath(os.path.join(_HERE, "..", "mhc"))
sys.path.insert(0, _HERE)     # testing.py
sys.path.insert(0, _MHC_DIR)  # mhc_pre_clamp_sinkhorn.py

# Marker printed by a child process so the driver can find its JSON result.
_MARK = "@@MHC_PRE_LV_RESULT@@"

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
_DEFAULT_SHAPE = (1, 256, 4, 3584)  # (B, S, N, D); N must be 4

_SWEEP_SHAPES = [
    (1, 64, 4, 3584),
    (1, 256, 4, 3584),
    (2, 1024, 4, 3584),
    (1, 4096, 4, 3584),
    (1, 1024, 4, 512),
    (1, 1024, 4, 1024),
    (1, 1024, 4, 2048),
]


def _resolve_shapes(args):
    if args.sweep:
        return list(_SWEEP_SHAPES)
    return [(args.B, args.S, args.N, args.D)]


def _kernel_kwargs(args):
    """Forward kwargs of mhc_pre_clamp_sinkhorn (not of its reference)."""
    return dict(
        norm_eps=args.norm_eps,
        hc_eps=args.hc_eps,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        iter_times=args.iter_times,
        need_backward=args.need_backward,
    )


def _ref_kwargs(args):
    """Forward kwargs of mhc_pre_clamp_sinkhorn_ref (no need_backward)."""
    return dict(
        norm_eps=args.norm_eps,
        hc_eps=args.hc_eps,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        iter_times=args.iter_times,
    )


# ---------------------------------------------------------------------------
# Child: single-variant benchmark
# ---------------------------------------------------------------------------


def _apply_variant_env(variant: str) -> None:
    """Set the pass toggle and a fresh per-run cache dir. Must run before the
    first triton import/compile.

    A brand-new cache dir guarantees both variants recompile from scratch, so a
    stale cache can never hide the pass's effect (triton's cache key does not
    include the pass implementation)."""
    if variant == "off":
        os.environ["TRITON_DISABLE_LANE_VECTORIZE"] = "1"
    else:
        os.environ.pop("TRITON_DISABLE_LANE_VECTORIZE", None)
    cache_dir = tempfile.mkdtemp(prefix=f"triton_bench_mhc_pre_lv_{variant}_")
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    print(f"[LV {variant}] TRITON_CACHE_DIR={cache_dir}", file=sys.stderr, flush=True)


def _device(torch):
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cuda"


def _sync(device: str, torch) -> None:
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _make_inputs(torch, B, S, N, D, dtype, device):
    """Match test_mhc_pre_clamp_sinkhorn._make_inputs."""
    hc_mix = N * (N + 2)  # 24 for N == 4
    hc_d = N * D
    x = torch.randn(B, S, N, D, dtype=dtype, device=device)
    phi = torch.randn(hc_mix, hc_d, dtype=torch.float32, device=device)
    alpha = torch.randn(3, dtype=torch.float32, device=device)
    base = torch.randn(hc_mix, dtype=torch.float32, device=device)
    return x, phi, alpha, base


def _bytes_io(B, S, N, D, dtype):
    """Read x + phi + alpha + base, write y + post_out + comb_frag."""
    elem = dtype.itemsize
    hc_mix = N * (N + 2)
    hc_d = N * D
    T = B * S
    read = (T * N * D * elem      # x
            + hc_mix * hc_d * 4   # phi (fp32)
            + 3 * 4               # alpha
            + hc_mix * 4)         # base
    write = (T * D * elem         # y
             + T * N * 4          # post_out
             + T * N * N * 4)     # comb_frag
    return read + write


def _bench_wall(fn, device, torch, warmup, rep):
    for _ in range(warmup):
        fn()
    _sync(device, torch)
    times = []
    for _ in range(rep):
        _sync(device, torch)
        t0 = time.perf_counter()
        fn()
        _sync(device, torch)
        times.append((time.perf_counter() - t0) * 1e3)
    s = sorted(times)
    return dict(ms=s[len(s) // 2], min=s[0], max=s[-1], mean=sum(s) / len(s))


def _bench_kernel(fn, warmup, rep, do_bench_npu):
    avg = do_bench_npu([fn], warmup=warmup, active=rep, clear_l2_cache=True, keep_res=False)
    if isinstance(avg, list):
        avg = avg[0]
    return dict(ms=avg, min=avg, max=avg, mean=avg)


def _check_outputs(torch, out, ref):
    """Compare y / post_out / comb_frag (tolerances from test_mhc_pre_clamp_sinkhorn)."""
    y_ok = bool(torch.allclose(out["y"].float(), ref["y"].float(), atol=1e-2, rtol=1e-2))
    post_ok = bool(torch.allclose(out["post_out"].float(), ref["post_out"].float(), atol=1e-4, rtol=1e-4))
    comb_ok = bool(torch.allclose(out["comb_frag"].float(), ref["comb_frag"].float(), atol=1e-4, rtol=1e-4))
    return y_ok and post_ok and comb_ok, dict(y=y_ok, post_out=post_ok, comb_frag=comb_ok)


def _run_child(args) -> int:
    _apply_variant_env(args.variant)

    import torch  # noqa: E402  (import after env is set)
    from mhc_pre_clamp_sinkhorn import mhc_pre_clamp_sinkhorn, mhc_pre_clamp_sinkhorn_ref  # noqa: E402
    from testing import do_bench_npu  # noqa: E402

    device = _device(torch)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    kwargs = _kernel_kwargs(args)
    ref_kwargs = _ref_kwargs(args)

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    log(f"[LV {args.variant}] device={device} dtype={args.dtype} mode={args.mode} "
        f"iter_times={args.iter_times} clamp=({args.clamp_min},{args.clamp_max}) "
        f"need_backward={args.need_backward}")

    results = []
    for (B, S, N, D) in _resolve_shapes(args):
        label = f"B{B}_S{S}_N{N}_D{D}"
        torch.manual_seed(0)
        x, phi, alpha, base = _make_inputs(torch, B, S, N, D, dtype, device)

        fn = lambda: mhc_pre_clamp_sinkhorn(x, phi, alpha, base, **kwargs)

        check = None
        detail = None
        if args.check:
            ref = mhc_pre_clamp_sinkhorn_ref(x, phi, alpha, base, **ref_kwargs)
            out = fn()
            check, detail = _check_outputs(torch, out, ref)
            log(f"[LV {args.variant}] {label}: check={'PASS' if check else 'FAIL'} {detail}")

        if args.mode == "kernel":
            t = _bench_kernel(fn, args.warmup, args.rep, do_bench_npu)
        else:
            t = _bench_wall(fn, device, torch, args.warmup, args.rep)

        bw = _bytes_io(B, S, N, D, dtype) / (t["ms"] * 1e-3) / 1e9
        log(f"[LV {args.variant}] {label}: {t['ms']:.4f} ms  {bw:.2f} GB/s")
        results.append(dict(label=label, B=B, S=S, N=N, D=D,
                            ms=t["ms"], min=t["min"], max=t["max"], mean=t["mean"],
                            bw_gbs=bw, check=check, detail=detail))

    payload = dict(variant=args.variant, device=device, mode=args.mode, dtype=args.dtype,
                   iter_times=args.iter_times, clamp_min=args.clamp_min,
                   clamp_max=args.clamp_max, need_backward=args.need_backward,
                   results=results)
    print(_MARK + json.dumps(payload), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Driver: run both variants and compare
# ---------------------------------------------------------------------------


def _parse_marker(stdout: str):
    for line in stdout.splitlines():
        if line.startswith(_MARK):
            try:
                return json.loads(line[len(_MARK):])
            except json.JSONDecodeError:
                return None
    return None


def _run_driver(args) -> int:
    payloads = {}
    for variant in ("on", "off"):
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--variant", variant,
            "--mode", args.mode,
            "--dtype", args.dtype,
            "--warmup", str(args.warmup),
            "--rep", str(args.rep),
            "--B", str(args.B), "--S", str(args.S),
            "--N", str(args.N), "--D", str(args.D),
            "--iter-times", str(args.iter_times),
            "--norm-eps", str(args.norm_eps),
            "--hc-eps", str(args.hc_eps),
            "--clamp-min", str(args.clamp_min),
            "--clamp-max", str(args.clamp_max),
        ]
        if args.sweep:
            cmd.append("--sweep")
        if args.check:
            cmd.append("--check")
        if args.need_backward:
            cmd.append("--need-backward")

        print(f"\n=== running variant: LV {variant} (pass {'enabled' if variant == 'on' else 'disabled'}) ===")
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        parsed = _parse_marker(proc.stdout)
        if parsed is None:
            print(proc.stdout)
            print(f"ERROR: variant LV {variant} produced no result (exit {proc.returncode})")
            return 1
        payloads[variant] = parsed

    return 0 if _print_comparison(payloads, args) else 1


def _index_by_label(payload):
    return {r["label"]: r for r in payload["results"]}


def _print_comparison(payloads, args):
    on = _index_by_label(payloads["on"])
    off = _index_by_label(payloads["off"])
    mode = payloads["on"]["mode"]
    tag = "avg" if mode == "kernel" else "median"

    print()
    print(f"Device : {payloads['on']['device']}   timing={mode}   dtype={args.dtype}   "
          f"iter_times={args.iter_times}   clamp=({args.clamp_min},{args.clamp_max})   "
          f"need_backward={args.need_backward}")
    print(f"Warmup : {args.warmup}   Rep/Active: {args.rep}")
    print()

    hdr = (f"{'config':>22} | {'LV on(ms)':>10} {'LV off(ms)':>11} "
           f"{'speedup':>8} | {'BW on':>8} {'BW off':>8} | {'check':>5}")
    print(hdr)
    print("-" * len(hdr))

    speedups = []
    for label in on:
        r_on, r_off = on[label], off.get(label)
        if r_off is None:
            continue
        sp = r_off["ms"] / r_on["ms"] if r_on["ms"] > 0 else float("inf")
        speedups.append(sp)
        check = "" if r_on["check"] is None else ("ok" if r_on["check"] else "FAIL")
        direction = "up" if sp > 1 else "dn"
        print(f"{label:>22} | {r_on['ms']:>10.4f} {r_off['ms']:>11.4f} "
              f"{sp:>7.2f}{direction} | {r_on['bw_gbs']:>7.1f} {r_off['bw_gbs']:>8.1f} | {check:>5}")

    if speedups:
        avg = sum(speedups) / len(speedups)
        print("-" * len(hdr))
        print(f"{'mean speedup (off/on)':>22} | {avg:>7.3f}x "
              f"(>1 means the lane-vectorize pass helps)")
    else:
        print("(no comparable results)")

    # Surface any failed correctness checks from either variant.
    ok = True
    for variant in ("on", "off"):
        bad = [r["label"] for r in payloads[variant]["results"] if r["check"] is False]
        if bad:
            print(f"WARNING: LV {variant} correctness FAILED for: {', '.join(bad)}")
            ok = False

    print()
    print(f"Note: {tag} latency reported; speedup = LV_off / LV_on.")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark mhc_pre_clamp_sinkhorn with/without the triton-lane-vectorize pass")
    # internal: select a single variant (child mode); normally unset
    p.add_argument("--variant", choices=["on", "off"], default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--B", type=int, default=_DEFAULT_SHAPE[0], help="batch size")
    p.add_argument("--S", type=int, default=_DEFAULT_SHAPE[1], help="sequence length")
    p.add_argument("--N", type=int, default=_DEFAULT_SHAPE[2], help="head multiplier (must be 4)")
    p.add_argument("--D", type=int, default=_DEFAULT_SHAPE[3], help="hidden dim per head")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="input dtype for x/y")
    p.add_argument("--mode", choices=["wall", "kernel"], default="wall",
                   help="wall: perf_counter+sync; kernel: do_bench_npu (mspti/profiler)")
    p.add_argument("--iter-times", type=int, default=20, help="sinkhorn iteration count")
    p.add_argument("--norm-eps", type=float, default=1e-6, help="RMSNorm epsilon")
    p.add_argument("--hc-eps", type=float, default=1e-6, help="sinkhorn epsilon")
    p.add_argument("--clamp-min", type=float, default=0.0, help="logits clamp min (0 = disabled)")
    p.add_argument("--clamp-max", type=float, default=0.0, help="logits clamp max (0 = disabled)")
    p.add_argument("--need-backward", action="store_true",
                   help="request the extra saved intermediates")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("--rep", type=int, default=50, help="measurement iterations (active runs for kernel mode)")
    p.add_argument("--check", action="store_true", help="verify output against mhc_pre_clamp_sinkhorn_ref")
    p.add_argument("--sweep", action="store_true", help="benchmark a set of shapes instead of one")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.variant is not None:
        return _run_child(args)
    return _run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
