"""Chunked (delta-rule) KDA: reference algebra + the numerics report we need first.

Why this exists
---------------
The serial kernel is reduce-bound (4 cross-lane reductions per token, ~0.8 us of
latency each token; see KDA_ASCENDC_VS_TLE.md 6.5).  The way out is the chunked
delta-rule form, which turns those reductions into cube matmuls.  The cube probe
says that is feasible (~7.4 TFLOP/s on one core for a dependent 128^3 chain,
bf16 in / fp32 accumulate, exact w.r.t. the bf16-rounded inputs).

Before writing that kernel there is one thing that can invalidate the whole
approach: the intra-chunk matrix needs the *relative* decay

    A[t,s] = sum_j k_s[j] * exp(gc_t[j] - gc_s[j]) * k_t[j],      s <= t

and the only kernel-expressible factorisation is

    A = (k * exp(-gc)) @ (x * exp(gc))^T

where exp(gc) <= 1 but exp(-gc) can be astronomically large.  KDA's gate is
lower_bound * sigmoid(...) with lower_bound < 0, so per-token log-decay can be
as low as -5, and exp(+5C) overflows fp32 by C ~ 18.  Chunk size is therefore
set by numerics, not by taste, and that is what this script measures: the
factored form against the explicit O(C^2 K) one, per chunk size and dtype.

It also checks the algebra itself (chunked fp64 vs the serial loop) so that a
later Triton bug cannot be mistaken for a derivation bug, and prints the gate
statistics -- if the deployed gates decay as fast as the synthetic bench inputs
do, the recurrence has an effective memory of a token or two, which changes both
the accuracy budget and what the kernel is even for.

    python3 chunked_ref.py                       # default sweep, CPU, seconds
    python3 chunked_ref.py --tokens 512 --chunks 8,16,32,64
    python3 chunked_ref.py --dtypes fp32,bf16,fp16
"""

from __future__ import annotations

import argparse

import torch


# ---------------------------------------------------------------------------
# The two matmuls the recurrence needs inside a chunk
# ---------------------------------------------------------------------------
def relative_gram_explicit(kn: torch.Tensor, x: torch.Tensor,
                           gc: torch.Tensor) -> torch.Tensor:
    """A[t,s] = sum_j k_s[j] * exp(gc_t[j] - gc_s[j]) * x_t[j], for s <= t.

    Written as the definition, O(C^2 K), and numerically safe because
    exp(gc_t - gc_s) <= 1 for s <= t.  This is the ground truth the factored
    form is compared against; a kernel cannot afford it.
    """
    C = kn.shape[0]
    out = kn.new_zeros(C, C, kn.shape[1])
    for t in range(C):
        decay = torch.exp(gc[t][None, :, :] - gc[: t + 1])   # [t+1,H,K], <= 1
        out[t, : t + 1] = (kn[: t + 1] * decay * x[t][None]).sum(-1)
    return out


def relative_gram_factored(kn: torch.Tensor, x: torch.Tensor, gc: torch.Tensor,
                           dtype: torch.dtype) -> torch.Tensor:
    """The only kernel-expressible form: (k*exp(-gc)) @ (x*exp(gc))^T.

    Two matmuls on the cube instead of C^2 reductions, at the cost of an
    exp(-gc) factor whose range explodes with C.  Returned in fp32.
    """
    e = torch.exp(gc)
    r = torch.exp(-gc)
    lhs = (kn * r).to(dtype)
    rhs = (x * e).to(dtype)
    return torch.einsum("shk,thk->tsh", lhs, rhs).float()


# ---------------------------------------------------------------------------
# Chunked recurrence
# ---------------------------------------------------------------------------
def chunked_ref(qn: torch.Tensor, kn: torch.Tensor, v: torch.Tensor,
                decay: torch.Tensor, beta: torch.Tensor, S0: torch.Tensor,
                C: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked delta-rule KDA, one head-batch, explicit intra-chunk matmuls.

    qn, kn: [T,H,K]  v: [T,H,V]  decay: [T,H,K] (= exp(gate))  beta: [T,H]
    S0:     [H,K,V]  (state before the first token)

    Per chunk, with gc = cumsum(log decay) and Lam = exp(gc) (<= 1):

        M   = I + strict_tril(diag(beta) A)          unit lower triangular
        u   = M^-1 diag(beta) (v - (kn*Lam) S0)
        out = (qn*Lam) S0 + Aq u
        S'  = Lam[-1] * S0 + sum_s (kn_s * exp(gc[-1]-gc_s)) u_s

    which is the serial recurrence rewritten; `chunked_ref` with C >= T is the
    serial loop with extra steps, and the report checks exactly that.
    """
    T, H, K = qn.shape
    V = v.shape[-1]
    out = qn.new_zeros(T, H, V)
    S = S0.clone()
    g = torch.log(decay)

    for c0 in range(0, T, C):
        c1 = min(c0 + C, T)
        n = c1 - c0
        sl = slice(c0, c1)
        eye = torch.eye(n, dtype=qn.dtype, device=qn.device).unsqueeze(-1)

        gc = torch.cumsum(g[sl], dim=0)              # [n,H,K], <= 0
        lam = torch.exp(gc)                          # [n,H,K], <= 1

        A = relative_gram_explicit(kn[sl], kn[sl], gc)    # [n,n,H]
        Aq = relative_gram_explicit(kn[sl], qn[sl], gc)
        strict = torch.tril(A.permute(2, 0, 1), -1).permute(1, 2, 0)

        # Row scaling, not column: u_t = beta_t (v_t - sum_{s<t} A[t,s] u_s), so
        # the coefficient is beta of the *row* token.  beta[sl][None] puts it on
        # the column index and silently solves a different system.
        M = eye + beta[sl][:, None, :] * strict      # [n,n,H]
        rhs = beta[sl][:, :, None] * (
            v[sl] - torch.einsum("nhk,hkv->nhv", kn[sl] * lam, S)
        )
        u = torch.linalg.solve_triangular(
            M.permute(2, 0, 1), rhs.permute(1, 0, 2),
            upper=False, unitriangular=True,
        ).permute(1, 0, 2)                           # [n,H,V]

        out[sl] = (torch.einsum("nhk,hkv->nhv", qn[sl] * lam, S)
                   + torch.einsum("tsh,shv->thv", Aq, u))

        kbar = kn[sl] * torch.exp(gc[-1][None] - gc)      # [n,H,K], <= 1
        S = lam[-1][:, :, None] * S + torch.einsum("nhk,nhv->hkv", kbar, u)

    return out, S


def serial_ref(qn: torch.Tensor, kn: torch.Tensor, v: torch.Tensor,
               decay: torch.Tensor, beta: torch.Tensor,
               S0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The recurrence as written in the kernel: one token at a time."""
    T, H, K = qn.shape
    V = v.shape[-1]
    out = qn.new_zeros(T, H, V)
    S = S0.clone()
    for t in range(T):
        S = S * decay[t][:, :, None]
        vv = (v[t] - (S * kn[t][:, :, None]).sum(1)) * beta[t][:, None]
        S = S + kn[t][:, :, None] * vv[:, None, :]
        out[t] = (S * qn[t][:, :, None]).sum(1)
    return out, S


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_inputs(tokens: int, heads: int, dim: int, seed: int = 7,
                dtype: torch.dtype = torch.float64, device: str = "cpu",
                gate_gain: float = 1.0) -> dict:
    """Prepared qn/kn/decay/beta plus an initial state, in `dtype`.

    gate_gain scales the gate input before the sigmoid.  1.0 is the bench
    distribution; 0.0 makes every (token, key) share the same decay
    exp(-2.5) = 0.082 (the strongest *uniform* decay the bound allows); values
    well above 1 push the sigmoid towards 1 and the decay towards exp(-5),
    which is the stress case for the intra-chunk dynamic range.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    K = V = dim
    a = torch.randn(tokens, heads, K, generator=g) * 0.2 * gate_gain
    dt_bias = torch.randn(heads, K, generator=g) * 0.1 * gate_gain
    A_log = torch.randn(heads, generator=g) * 0.1
    lower_bound = -5.0
    scale = K**-0.5

    q = torch.randn(tokens, heads, K, generator=g) * 0.2
    k = torch.randn(tokens, heads, K, generator=g) * 0.2
    v = torch.randn(tokens, heads, V, generator=g) * 0.2
    b = torch.randn(tokens, heads, generator=g)

    decay_base = torch.exp(A_log)[None, :, None]        # broadcast over T and K
    gate = lower_bound / (1.0 + torch.exp(-(decay_base * (a + dt_bias))))
    decay = torch.exp(gate)
    beta = torch.sigmoid(b)
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    S0 = torch.randn(heads, K, V, generator=g) * 0.1
    return dict(qn=qn.to(dtype), kn=kn.to(dtype), v=v.to(dtype),
                decay=decay.to(dtype), beta=beta.to(dtype), S0=S0.to(dtype))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    den = b.abs().max().item()
    return (a - b).abs().max().item() / den if den > 0 else float("inf")


def _lower_only(x: torch.Tensor) -> torch.Tensor:
    """Keep s <= t.  The factored gemm produces a full C x C matrix, and its
    upper half is exp(gc_t - gc_s) with t < s, i.e. > 1 and astronomically so;
    the algorithm never reads it, so comparing it against the explicit form
    (which leaves it zero) measures nothing but the overflow."""
    C = x.shape[0]
    keep = torch.tril(torch.ones(C, C, dtype=torch.bool,
                                 device=x.device)).unsqueeze(-1)
    return torch.where(keep, x, torch.zeros_like(x))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--chunks", default="8,16,32,64")
    p.add_argument("--dtypes", default="fp32,bf16")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gate-gain", type=float, default=1.0,
                   help="scales the gate input: 1.0 = bench-like (fast decay), "
                        "negative values approach decay=1 (slow, long memory)")
    args = p.parse_args()

    dt_map = {"fp32": torch.float32, "bf16": torch.bfloat16,
              "fp16": torch.float16}
    chunks = [int(x) for x in args.chunks.split(",")]
    dtypes = [dt_map[d.strip()] for d in args.dtypes.split(",")]
    tok = args.tokens

    base = make_inputs(tok, args.heads, args.dim, args.seed, torch.float64,
                      gate_gain=args.gate_gain)
    qn, kn = base["qn"], base["kn"]
    decay, beta, S0 = base["decay"], base["beta"], base["S0"]

    # --- gate statistics ---------------------------------------------------
    g = torch.log(decay)
    print(f"tokens={tok} heads={args.heads} dim={args.dim} seed={args.seed} "
          f"gate_gain={args.gate_gain}")
    print("gate statistics (fp64):")
    print(f"  decay   min {decay.min():.4e}  median {decay.median():.4e}  "
          f"mean {decay.mean():.4e}")
    frac = (decay < 0.5).double().mean().item()
    print(f"  decay < 0.5 on {100 * frac:.1f}% of (token,key) entries")
    print(f"  per-token log-decay: min {g.min():.3f}  mean {g.mean():.3f}")
    for C in chunks:
        gc = torch.cumsum(g[: (tok // C) * C].reshape(-1, C, *g.shape[1:]), dim=1)
        print(f"  chunk C={C:3d}: max exp(-gc) = {torch.exp(-gc.min()):.3e}"
              f"   (fp32 max 3.4e38)")

    # --- algebra check -----------------------------------------------------
    print("\nchunked vs serial, fp64 (validates the derivation, not the kernel):")
    ref_out, ref_S = serial_ref(qn, kn, base["v"], decay, beta, S0)
    for C in chunks:
        out, S = chunked_ref(qn, kn, base["v"], decay, beta, S0, C)
        print(f"  C={C:3d}  out rel {_rel(out, ref_out):.3e}   "
              f"state rel {_rel(S, ref_S):.3e}")

    # --- factored form, i.e. what a kernel can compute ----------------------
    print("\nfactored vs explicit intra-chunk gram (kernel-expressible form):")
    print("  compared on the lower triangle only, which is all the algorithm reads")
    print(f"  {'C':>4} {'dtype':>6} {'max exp(-gc)':>13} {'A rel':>10} {'Aq rel':>10}"
          f" {'|A| max':>10}  note")
    for C in chunks:
        gc_all = torch.cumsum(
            g[: (tok // C) * C].reshape(-1, C, *g.shape[1:]), dim=1)
        gc, rmax = gc_all[0], torch.exp(-gc_all.min()).item()
        kn_c, qn_c = kn[:C], qn[:C]
        A_ref = _lower_only(relative_gram_explicit(kn_c, kn_c, gc))
        Aq_ref = _lower_only(relative_gram_explicit(kn_c, qn_c, gc))
        for dt in dtypes:
            A_f = relative_gram_factored(kn_c, kn_c, gc, dt)
            Aq_f = relative_gram_factored(kn_c, qn_c, gc, dt)
            name = {torch.float32: "fp32", torch.bfloat16: "bf16",
                    torch.float16: "fp16"}[dt]
            amax = A_f.abs().max().item()
            note = ""
            if not torch.isfinite(A_f).all() or not torch.isfinite(Aq_f).all():
                note = "non-finite -> chunk too big for this dtype"
            print(f"  {C:4d} {name:>6} {rmax:13.3e} "
                  f"{_rel(_lower_only(A_f), A_ref):10.3e} "
                  f"{_rel(_lower_only(Aq_f), Aq_ref):10.3e} {amax:10.3e}  {note}")

    print("\nhow to read this:")
    print("  algebra rows ~1e-15  : the derivation is right; a later mismatch is kernel code")
    print("  A/Aq rel err        : the accuracy bill for the factored form, lower triangle")
    print("                        only (that is all the algorithm reads)")
    print("  |A| max (unmasked)  : grows like exp(+gc) in the upper half; when it is non-")
    print("                        finite the factor overflowed the dtype and even the valid")
    print("                        half is garbage, so that (C, dtype) is simply unusable")
    print("  max exp(-gc)        : the factor that has to survive. Under 1e30 leaves room;")
    print("                        past ~1e38 it is inf in both fp32 and bf16")


if __name__ == "__main__":
    main()
