"""Self-contained benchmark / correctness harness for the KDA Triton kernels.

Runs on the target NPU, needs ``torch``, ``torch_npu`` and the FlagTree
``triton`` — no sglang, no ``benchmark`` framework.  Three subcommands:

    calib   grid sweep of a trivial kernel.  Answers "how many vector cores
            does this machine actually schedule in parallel", which every
            BV / grid decision in v1 depends on (KDA_CORES).
    sweep   latency sweep, v0 against a matrix of v1 configurations, at the
            prefill shapes that matter (H=4/8, T=1k..16k).
    check   numerical comparison against a plain serial PyTorch reference.

Examples:

    # from test/CommonIR/agent_op/target/ (v0 and v1 live there)
    python3 bench_kda_tle.py calib
    python3 bench_kda_tle.py sweep --out sweep.json
    python3 bench_kda_tle.py check --out check.json

    # or against another checkout of v0
    python3 bench_kda_tle.py sweep --v0-dir /path/to/flagtree-ops/.../target

Every v1 knob is an environment variable, so a run can A/B a configuration
without editing code:

    KDA_CORES KDA_BV KDA_MAX_BV KDA_PREFETCH KDA_FLAT_STATE
    KDA_STAGES KDA_MULTIBUFFER KDA_NUM_WARPS
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

try:  # registers the "npu" backend; absent on a CPU-only box
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except ImportError:  # pragma: no cover
    _HAS_NPU = False

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

D = 128
SCALE = D**-0.5
LOWER_BOUND = -5.0

# Tolerances for `check`.  Outputs are bf16 and |out| runs ~1e-4 with a ~1e-2
# tail (q and k are L2-normalized before the scale), so atol has to stay well
# under that or the comparison passes everything on its own; the real bar is
# rtol.  v1 is deliberately *not* bit-exact (see the module docstring), so this
# is a tolerance check, not an equality check.
RTOL = 3e-2
ATOL = 2e-3


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------
def load_modules(v0_dir: str | None):
    """Import v0 and v1.  v0 is in this same directory by default; the env
    var exists so a run can point at another checkout (flagtree-ops)."""
    if v0_dir:
        os.environ["GLM53_KDA_V0_DIR"] = v0_dir
    os.environ.setdefault("GLM53_KDA_V0_DIR", HERE)
    import glm_kda_varlen_recurrent_npu_v0 as v0
    import glm_kda_varlen_recurrent_npu_v1 as v1

    return v0, v1


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def native_state(slots: int, heads: int, device: str) -> torch.Tensor:
    """[slots,H,V,K] contiguous, viewed as [slots,H,K,V] — the production pool.

    Allocating then transposing is the only way to get stride(2)==1,
    stride(3)==128, which is what the kernels index against.
    """
    pool = torch.randn(slots, heads, D, D, device=device) * 0.1
    return pool.transpose(-1, -2)


def clone_state(state: torch.Tensor) -> torch.Tensor:
    """Copy preserving the [V,K] backing (a plain .clone() makes it contiguous)."""
    return state.transpose(-1, -2).clone().transpose(-1, -2)


def make_case(heads: int, tokens: int, device: str, seed: int = 1979) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed + heads + tokens)
    tensors = {
        name: (torch.randn(1, tokens, heads, D, generator=g) * 0.2).to(device).bfloat16()
        for name in ("q", "k", "v", "a")
    }
    return dict(
        **tensors,
        b=torch.randn(1, tokens, heads, generator=g).to(device).bfloat16(),
        A_log=(torch.randn(heads, generator=g) * 0.1).to(device),
        dt_bias=(torch.randn(heads * D, generator=g) * 0.1).to(device),
        # Slot 0 is the padding row and is never written.
        state=native_state(2, heads, device),
        indices=torch.tensor([1], dtype=torch.int32, device=device),
        cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=device),
        lower_bound=LOWER_BOUND,
        scale=SCALE,
        prefill=True,
    )


def arms(case: dict) -> dict:
    """The kwargs every arm takes; `initial_state_source` is swapped per run."""
    return dict(
        q=case["q"], k=case["k"], v=case["v"], a=case["a"], b=case["b"],
        A_log=case["A_log"], dt_bias=case["dt_bias"],
        initial_state_indices=case["indices"], cu_seqlens=case["cu_seqlens"],
        lower_bound=case["lower_bound"], scale=case["scale"],
        prefill=case["prefill"],
    )


# ---------------------------------------------------------------------------
# Serial reference
# ---------------------------------------------------------------------------
def torch_reference(case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight per-token loop in fp32.  Slow, but independent of both kernels.

    Follows the documented semantics exactly:

        decay = exp(A_log); g = lower_bound * sigmoid(decay * (a + dt_bias))
        beta  = sigmoid(b); q = q/sqrt(sum(q*q)+1e-6)*scale; k likewise
        h *= exp(g); v -= sum(h*k, axis=K); v *= beta; h += outer(v, k)
        out = sum(h*q, axis=K)

    Indexing is logical ([head, K, V]), so it does not care which physical
    layout the state pool uses.
    """
    q, k, v, a, b = (case[n][0].float() for n in ("q", "k", "v", "a", "b"))
    A_log = case["A_log"].float()
    dt_bias = case["dt_bias"].float().view(-1, D)
    scale = float(case["scale"])
    lower_bound = float(case["lower_bound"])
    starts = case["cu_seqlens"].tolist()
    state = clone_state(case["state"]).float()
    heads = q.shape[1]

    decay = torch.exp(A_log)                       # [H]
    out = torch.zeros_like(v)
    for n in range(len(starts) - 1):
        bos, eos = int(starts[n]), int(starts[n + 1])
        slot = int(case["indices"][n])
        h = state[slot].clone()                    # [H, K, V]
        for t in range(bos, eos):
            g = lower_bound * torch.sigmoid(
                decay[:, None] * (a[t] + dt_bias)
            )                                       # [H, K]
            beta = torch.sigmoid(b[t])              # [H]
            qn = q[t] / torch.sqrt((q[t] * q[t]).sum(-1, keepdim=True) + 1e-6)
            kn = k[t] / torch.sqrt((k[t] * k[t]).sum(-1, keepdim=True) + 1e-6)
            qn = qn * scale
            h = h * torch.exp(g)[:, :, None]
            vv = (v[t] - (h * kn[:, :, None]).sum(1)) * beta[:, None]
            h = h + kn[:, :, None] * vv[:, None, :]
            out[t] = (h * qn[:, :, None]).sum(1).to(out.dtype)
        if slot > 0:
            state[slot] = h
    return out, state


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def timeit(run, state0, warmup: int, reps: int) -> dict:
    for _ in range(warmup):
        run(clone_state(state0))
    torch.npu.synchronize()

    times = []
    for _ in range(reps):
        state = clone_state(state0)      # outside the timed region
        torch.npu.synchronize()
        t0 = time.perf_counter()
        run(state)
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "min_ms": times[0],
        "p90_ms": times[min(len(times) - 1, int(len(times) * 0.9))],
        "n": len(times),
    }


# The ambient KDA_CORES survives apply_env(), so a shell-level
# `KDA_CORES=24 python3 bench_kda_tle.py sweep` still reaches the kernel.
_AMBIENT_CORES = os.environ.get("KDA_CORES")


def apply_env(env: dict) -> None:
    for key in (
        "KDA_CORES", "KDA_BV", "KDA_MAX_BV", "KDA_PREFETCH",
        "KDA_FLAT_STATE", "KDA_STAGES", "KDA_MULTIBUFFER", "KDA_NUM_WARPS",
    ):
        os.environ.pop(key, None)
    if _AMBIENT_CORES is not None:
        os.environ["KDA_CORES"] = _AMBIENT_CORES
    for key, value in env.items():
        os.environ[key] = str(value)


# ---------------------------------------------------------------------------
# calib
# ---------------------------------------------------------------------------
def cmd_calib(args, v1) -> None:
    import triton
    import triton.language as tl

    # Fixed work per block, growing grid: the latency is flat while the grid
    # fits in one wave and steps up by one block-time per extra wave, so the
    # first step is the number of vector cores the scheduler gives us.
    @triton.jit
    def _probe(X, Y, ITER: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + off)
        for _ in tl.static_range(ITER):
            x = x * 1.0000001 + 1.0
        tl.store(Y + off, x)

    device = args.device
    block = 8192
    iters = 128
    grids = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]
    rows = []
    print("grid  median_us  us_per_wave")
    for g in grids:
        n = g * block
        x = torch.randn(n, device=device)
        y = torch.empty_like(x)
        run = lambda: _probe[(g,)](x, y, ITER=iters, BLOCK=block)  # noqa: E731
        for _ in range(10):
            run()
        torch.npu.synchronize()
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            run()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        med = statistics.median(ts)
        rows.append({"grid": g, "median_us": med})
        print(f"{g:5d}  {med:10.2f}")

    # Step detection: the first grid whose time is >10 % above the previous one
    # has started a second wave, so the previous grid is the wave width.
    base = min(r["median_us"] for r in rows)
    cores = rows[-1]["grid"]
    prev = rows[0]
    for row in rows[1:]:
        if row["median_us"] > prev["median_us"] * 1.10:
            cores = prev["grid"]
            break
        prev = row
    print(f"\nbaseline {base:.2f} us; first wave width ~= {cores}")
    print(f"-> pass KDA_CORES={cores} to v1")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "suggested_cores": cores}, fh, indent=2)
        print(f"wrote {args.out}")


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def sweep_configs() -> list[dict]:
    return [
        {"name": "v0_autotune", "impl": "v0", "env": {}},
        {"name": "v1_bv_auto", "impl": "v1", "env": {}},
        {"name": "v1_bv_auto_pf", "impl": "v1", "env": {"KDA_PREFETCH": 1}},
        {"name": "v1_bv_auto_st2", "impl": "v1", "env": {"KDA_STAGES": 2}},
        {"name": "v1_bv_auto_st3", "impl": "v1", "env": {"KDA_STAGES": 3}},
        {"name": "v1_bv_auto_mb", "impl": "v1", "env": {"KDA_MULTIBUFFER": 1}},
        {"name": "v1_bv_auto_flat", "impl": "v1", "env": {"KDA_FLAT_STATE": 1}},
        {"name": "v1_bv16", "impl": "v1", "env": {"KDA_BV": 16}},
        {"name": "v1_bv32", "impl": "v1", "env": {"KDA_BV": 32}},
        {"name": "v1_bv64", "impl": "v1", "env": {"KDA_BV": 64}},
        {"name": "v1_bv128", "impl": "v1", "env": {"KDA_BV": 128}},
    ]


def cmd_sweep(args, v0, v1) -> None:
    heads_list = [int(x) for x in args.heads.split(",")]
    tokens_list = [int(x) for x in args.tokens.split(",")]
    results = []

    print(f"cores={os.environ.get('KDA_CORES', '40')}  "
          f"warmup={args.warmup}  reps={args.reps}")
    header = f"{'case':>18}  " + "  ".join(f"{c['name']:>18}" for c in sweep_configs())
    print(header)
    print("-" * len(header))

    for heads in heads_list:
        for tokens in tokens_list:
            case = make_case(heads, tokens, args.device, seed=args.seed)
            state0 = clone_state(case["state"])
            kw = arms(case)
            row = {"heads": heads, "tokens": tokens, "arms": {}}
            cells = []
            for cfg in sweep_configs():
                apply_env(cfg["env"])
                if cfg["impl"] == "v0":
                    run = lambda st, kw=kw: v0.glm_kda_varlen_recurrent_npu(
                        initial_state_source=st, **kw
                    )
                else:
                    run = lambda st, kw=kw: v1.call_v1(
                        initial_state_source=st, **kw
                    )
                try:
                    stats = timeit(run, state0, args.warmup, args.reps)
                    stats["cfg"] = v1.config_summary() if cfg["impl"] == "v1" else None
                except Exception as exc:  # compile failure, hang guard, etc.
                    stats = {"error": f"{type(exc).__name__}: {exc}".splitlines()[0]}
                row["arms"][cfg["name"]] = stats
                cells.append(f"{stats.get('median_ms', float('nan')):18.4f}"
                             if "median_ms" in stats else f"{'ERR':>18}")
            results.append(row)
            print(f"{f'H{heads} T{tokens}':>18}  " + "  ".join(cells), flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"cores": os.environ.get("KDA_CORES", "40"),
                       "warmup": args.warmup, "reps": args.reps,
                       "rows": results}, fh, indent=2)
        print(f"\nwrote {args.out}")


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def cmd_check(args, v0, v1) -> None:
    heads_list = [int(x) for x in args.heads.split(",")]
    tokens_list = [int(x) for x in args.tokens.split(",")]
    results = []
    ok = True

    print(f"rtol={RTOL} atol={ATOL}")
    for heads in heads_list:
        for tokens in tokens_list:
            case = make_case(heads, tokens, args.device, seed=args.seed)
            ref_out, ref_state = torch_reference(case)
            kw = arms(case)

            row = {"heads": heads, "tokens": tokens}
            for name, fn in (("v0", v0.glm_kda_varlen_recurrent_npu),
                             ("v1", v1.call_v1)):
                state = clone_state(case["state"])
                out = fn(initial_state_source=state, **kw)
                torch.npu.synchronize()
                out_d = (out.float() - ref_out).abs()
                st_d = (state.float() - ref_state).abs()
                cos = torch.nn.functional.cosine_similarity(
                    out.float().flatten().unsqueeze(0),
                    ref_out.flatten().unsqueeze(0),
                ).item()
                passed = (
                    torch.allclose(out.float(), ref_out, rtol=RTOL, atol=ATOL)
                    and torch.allclose(state.float(), ref_state, rtol=RTOL, atol=ATOL)
                )
                row[name] = {
                    "out_max_abs": out_d.max().item(),
                    "state_max_abs": st_d.max().item(),
                    "cos": cos,
                    "passed": bool(passed),
                }
                ok = ok and passed
            results.append(row)
            print(f"H{heads} T{tokens:<6} "
                  f"v0 out={row['v0']['out_max_abs']:.3e} "
                  f"state={row['v0']['state_max_abs']:.3e} cos={row['v0']['cos']:.8f} "
                  f"{'PASS' if row['v0']['passed'] else 'FAIL'} | "
                  f"v1 out={row['v1']['out_max_abs']:.3e} "
                  f"state={row['v1']['state_max_abs']:.3e} cos={row['v1']['cos']:.8f} "
                  f"{'PASS' if row['v1']['passed'] else 'FAIL'}", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rtol": RTOL, "atol": ATOL, "rows": results}, fh, indent=2)
        print(f"\nwrote {args.out}")
    raise SystemExit(0 if ok else 1)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("calib", "sweep", "check"))
    parser.add_argument("--v0-dir", default=None,
                        help="directory holding glm_kda_varlen_recurrent_npu_v0.py")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", default="4,8")
    parser.add_argument("--tokens", default="1024,2048,4096,8192,16384")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1979)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.command == "check" and args.tokens == "1024,2048,4096,8192,16384":
        args.tokens = "64,128,256"      # the serial reference is per-token

    if not _HAS_NPU:
        raise SystemExit("torch_npu is not importable; this harness needs an NPU")
    torch.npu.set_device(args.device)
    v0, v1 = load_modules(args.v0_dir)

    if args.command == "calib":
        cmd_calib(args, v1)
    elif args.command == "sweep":
        cmd_sweep(args, v0, v1)
    else:
        cmd_check(args, v0, v1)


if __name__ == "__main__":
    main()
