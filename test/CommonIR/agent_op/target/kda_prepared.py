"""KDA prefill as two launches: batched prep, then the serial recurrence.

Why this exists
---------------
The measured budget of the fused kernel (KDA_ASCENDC_VS_TLE.md 6.5-6.7) is

    per token : 1082 ns total = 612 (recurrence) + 470 (prep)

and the prep's 470 ns is essentially all two cross-lane reductions per token
(s_q, s_k), each worth ~480 cycles. Those reductions are *token-local*, so they
do not have to sit in the serial token loop at all: a separate pass that
processes C tokens per program turns 2 reductions per token into 2 per chunk
(~1/8th of the work at C=16) and takes the whole prep dependency chain off the
recurrence's critical path.

Expected, and the only reason to do it: 612 + ~100 = ~712 ns/token, i.e. ~1.5x.
That is smaller than the chunked delta-rule route promised, but unlike that
route it does not depend on the gate distribution, which is the one input we do
not have:

    chunked C=16 (safe for any gate)   ~808 ns/token best case -> 1.34x
    chunked C=64 (needs slow gates)    ~420 ns/token            -> 2.6x, unknown
    serial fused (v1)                  1082 ns/token            -> 1.00x
    serial + batched prep               ~712 ns/token            -> ~1.5x

So this file is the unconditional half of the remaining work. The chunked kernel
(kda_chunked.py) stays in the tree, correct and parked, until someone can
measure what fraction of the deployed gates decay slowly.

Shape of the two kernels
------------------------
`_prep_kernel`   grid = (cdiv(T, C), H): q/k L2 norms, gate, beta, for C tokens
                 at a time.  One strided load per tensor, two row reductions
                 for the whole chunk, four strided stores.
`_recur_kernel`  grid = (D//BV, sequences, H): the same serial recurrence as
                 v1's optimized kernel, but reading the prepared fp32 tensors
                 instead of computing anything token-local.  Two reductions per
                 token remain (the state ones); those are inherent.

The scratch is ~3 x T*H*D fp32 (50 MB at T=8192, H=4), about 0.1 ms of DRAM
traffic at this device's rate, against the ~0.5 ms of reduction latency it
removes.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

try:  # registers the "npu" backend
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except ImportError:  # pragma: no cover
    _HAS_NPU = False

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pass 1: per-token inputs, C tokens at a time
# ---------------------------------------------------------------------------
@triton.jit
def _prep_kernel(
    Q, K, A, B, ALOG, BIAS,
    QN, KN, GD, BETA,
    T,
    scale, lower_bound,
    H: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)

    rows = chunk * C + tl.arange(0, C)
    live = rows < T
    off = rows[:, None] * (H * D) + head * D + tl.arange(0, D)[None, :]

    q = tl.load(Q + off, mask=live[:, None], other=0.0).to(tl.float32)
    k = tl.load(K + off, mask=live[:, None], other=0.0).to(tl.float32)
    a = tl.load(A + off, mask=live[:, None], other=0.0).to(tl.float32)
    b = tl.load(B + rows * H + head, mask=live, other=0.0).to(tl.float32)

    bias = tl.load(BIAS + head * D + tl.arange(0, D)).to(tl.float32)
    decay_base = tl.exp(tl.load(ALOG + head).to(tl.float32))

    # Both norms for the whole chunk: two reductions per C tokens instead of
    # two per token, and no serial dependency between them.
    s_q = tl.sum(q * q, axis=1)
    s_k = tl.sum(k * k, axis=1)
    qn = q * (scale / tl.sqrt(s_q + 1e-6))[:, None]
    kn = k * (1.0 / tl.sqrt(s_k + 1e-6))[:, None]

    gate = lower_bound / (1.0 + tl.exp(-(decay_base * (a + bias[None, :]))))
    gd = tl.exp(gate)
    beta = 1.0 / (1.0 + tl.exp(-b))

    tl.store(QN + off, qn, mask=live[:, None])
    tl.store(KN + off, kn, mask=live[:, None])
    tl.store(GD + off, gd, mask=live[:, None])
    tl.store(BETA + rows * H + head, beta, mask=live)


# ---------------------------------------------------------------------------
# Pass 2: the serial recurrence, prepared inputs
# ---------------------------------------------------------------------------
@triton.jit
def _recur_kernel(
    QN, KN, VV, GD, BETA,
    OUT, STATE, INDICES, STARTS,
    TRACK_INDICES, TRACK_LENS,
    TRACK_STATE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr,
):
    iv = tl.program_id(0)
    i_n = tl.program_id(1)
    head = tl.program_id(2)

    bos = tl.load(STARTS + i_n).to(tl.int64)
    eos = tl.load(STARTS + i_n + 1).to(tl.int64)
    seq_len = eos - bos

    ks = tl.arange(0, D)
    vs = iv * BV + tl.arange(0, BV)
    mask_k = ks < D
    mask_v = vs < D
    mask_h = mask_v[:, None] & mask_k[None, :]

    # State tile, kept in the pool's own [V,K] orientation so the load and the
    # store are both contiguous spans of BV*D floats.
    slot = tl.load(INDICES + i_n).to(tl.int64)
    pstate = slot * S0 + head * S1 + iv * (BV * D) + tl.arange(0, BV * D)
    state = tl.zeros([BV, D], tl.float32)
    if slot > 0:
        state = tl.reshape(tl.load(STATE + pstate), [BV, D])

    if TRACK_STATE:
        track_slot = tl.load(TRACK_INDICES + i_n).to(tl.int64)
        track_len = tl.load(TRACK_LENS + i_n)
        ptrack = (STATE + track_slot * S0 + head * S1
                  + ks[None, :] * 1 + vs[:, None] * D)
        if track_slot > 0 and track_len == 0:
            tl.store(ptrack, state, mask=mask_h)

    q_base = QN + head * D + ks
    k_base = KN + head * D + ks
    g_base = GD + head * D + ks
    v_base = VV + head * D + vs
    o_base = OUT + head * D + vs
    stride_t = H * D
    stride_t_b = H

    for i in range(seq_len):
        tcur = (bos + i) * stride_t
        row = bos + i

        qn = tl.load(q_base + tcur, mask=mask_k, other=0.0)
        kn = tl.load(k_base + tcur, mask=mask_k, other=0.0)
        decay = tl.load(g_base + tcur, mask=mask_k, other=1.0)
        v = tl.load(v_base + tcur, mask=mask_v, other=0.0).to(tl.float32)
        beta = tl.load(BETA + row * stride_t_b + head).to(tl.float32)

        # state *= decay; v = beta * (v - sum(state*k)); state += k v^T
        state = state * decay[None, :]
        v = v - tl.sum(state * kn[None, :], axis=1)
        v = v * beta
        state = state + kn[None, :] * v[:, None]

        if TRACK_STATE:
            if track_slot > 0 and i + 1 == track_len:
                tl.store(ptrack, state, mask=mask_h)

        out = tl.sum(state * qn[None, :], axis=1)
        tl.store(o_base + tcur, out.to(OUT.dtype.element_ty), mask=mask_v)

    if slot > 0:
        tl.store(STATE + pstate, tl.reshape(state, [BV * D]))


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------
def run_prepared(q, k, v, a, b, A_log, dt_bias, state, indices, cu_seqlens,
                 scale: float, lower_bound: float, BV: int = 16,
                 C: int = 16, track_indices=None, track_lens=None):
    """q,k,v,a: [T,H,D]; b: [T,H]; A_log: [H]; dt_bias: [H,D]; state mutated."""
    T, H, D = q.shape
    V = v.shape[-1]
    if state.stride(2) != 1 or state.stride(3) != D:
        raise ValueError("state must be K-contiguous (stride(2)==1)")

    f32 = torch.float32
    qn = torch.empty(T, H, D, dtype=f32, device=q.device)
    kn = torch.empty(T, H, D, dtype=f32, device=q.device)
    gd = torch.empty(T, H, D, dtype=f32, device=q.device)
    beta = torch.empty(T, H, dtype=f32, device=q.device)
    out = torch.empty_like(v)

    _prep_kernel[(triton.cdiv(T, C), H)](
        q, k, a, b, A_log, dt_bias, qn, kn, gd, beta, T,
        scale, lower_bound, H=H, D=D, C=C,
        num_warps=1, num_stages=1, multibuffer=False,
    )
    track = track_indices is not None
    _recur_kernel[(D // BV, indices.numel(), H)](
        qn, kn, v, gd, beta, out, state, indices, cu_seqlens,
        track_indices if track else indices,
        track_lens if track else cu_seqlens,
        TRACK_STATE=track,
        H=H, D=D, BV=BV,
        S0=state.stride(0), S1=state.stride(1),
        num_warps=1, num_stages=1, multibuffer=False,
    )
    return out


# ---------------------------------------------------------------------------
# Self-test: prep in torch, then serial in torch -- validates both kernels
# ---------------------------------------------------------------------------
def _torch_reference(q, k, v, a, b, A_log, dt_bias, S0, scale, lower_bound):
    # dt_bias arrives flat ([H*D], the same tensor the kernel reads as a pointer);
    # it has to be [H,D] to broadcast against a's [T,H,D].
    H, D = A_log.numel(), a.shape[-1]
    dt_bias = dt_bias.reshape(H, D)
    qn = q.float() / torch.sqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6) * scale
    kn = k.float() / torch.sqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)
    decay = torch.exp(lower_bound / (1.0 + torch.exp(
        -(torch.exp(A_log)[None, :, None] * (a.float() + dt_bias[None])))))
    beta = torch.sigmoid(b.float())
    T = q.shape[0]
    state = S0.clone()
    out = torch.zeros_like(v.float())
    for t in range(T):
        state = state * decay[t][:, :, None]
        vv = (v[t].float() - (state * kn[t][:, :, None]).sum(1)) * beta[t][:, None]
        state = state + kn[t][:, :, None] * vv[:, None, :]
        out[t] = (state * qn[t][:, :, None]).sum(1)
    return out, state


def _selftest(args) -> int:
    dev = args.device
    H, D = 2, args.dim
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    T = args.tokens

    def rnd(*shape, std=0.2):
        return (torch.randn(*shape, generator=g) * std).to(dev)

    q, k, v, a = (rnd(T, H, D).bfloat16() for _ in range(4))
    b = rnd(T, H).bfloat16()
    A_log = rnd(H, std=0.1)
    dt_bias = rnd(H, D, std=0.1).reshape(H * D)
    S0 = rnd(H, D, D, std=0.1)

    ref_out, ref_S = _torch_reference(q.cpu(), k.cpu(), v.cpu(), a.cpu(), b.cpu(),
                                      A_log.cpu(), dt_bias.cpu(), S0.cpu(),
                                      args.scale, args.lower_bound)
    pool = torch.zeros(2, H, D, D, dtype=torch.float32, device=dev)
    pool[1] = S0.transpose(-1, -2)
    state = pool.transpose(-1, -2)
    out = run_prepared(q, k, v, a, b, A_log, dt_bias.view(H, D), state,
                       torch.tensor([1], dtype=torch.int32, device=dev),
                       torch.tensor([0, T], dtype=torch.int32, device=dev),
                       args.scale, args.lower_bound, BV=args.bv, C=args.chunk)
    torch.npu.synchronize()

    out_rel = ((out.float().cpu() - ref_out).abs().max().item()
               / ref_out.abs().max().item())
    state_rel = ((state[1].float().cpu() - ref_S).abs().max().item()
                 / ref_S.abs().max().item())
    mag = out.float().abs().max().item() / ref_out.abs().max().item()
    ok = out_rel < args.tol and state_rel < args.tol
    print(f"T={T} H={H} C={args.chunk} BV={args.bv}  out rel {out_rel:.3e}  "
          f"state rel {state_rel:.3e}  |out| ratio {mag:9.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _bench(args) -> int:
    dev = args.device
    H, D = args.heads, args.dim
    T = args.tokens
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    def rnd(*shape, std=0.2):
        return (torch.randn(*shape, generator=g) * std).to(dev)

    q, k, v, a = (rnd(T, H, D).bfloat16() for _ in range(4))
    b = rnd(T, H).bfloat16()
    A_log = rnd(H, std=0.1)
    dt_bias = rnd(H, D, std=0.1)
    S0 = rnd(H, D, D, std=0.1)

    pool = torch.zeros(2, H, D, D, dtype=torch.float32, device=dev)
    pool[1] = S0.transpose(-1, -2)
    state0 = pool.transpose(-1, -2)
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)

    def clone(st):
        return st.transpose(-1, -2).clone().transpose(-1, -2)

    def run(st):
        return run_prepared(q, k, v, a, b, A_log, dt_bias, st, idx, qsl,
                            args.scale, args.lower_bound, BV=args.bv, C=args.chunk)

    for _ in range(args.warmup):
        run(clone(state0))
    torch.npu.synchronize()
    ts = []
    for _ in range(args.reps):
        st = clone(state0)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        run(st)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ms = statistics.median(ts)

    blocks = H * (D // args.bv)
    print(f"T={T} H={H} C={args.chunk} BV={args.bv} blocks={blocks} "
          f"reps={args.reps}")
    print(f"  {'variant':<22} {'ms':>8} {'ns/token':>9}")
    print(f"  {'two-launch prepared':<22} {ms:8.3f} {ms * 1e6 / T:9.1f}")
    for name, ref_ms in (("serial fused (v1)", 8.86), ("serial prep-ablated", 5.01)):
        if T == 8192:
            print(f"  {name + ' [earlier]':<22} {ref_ms:8.3f} "
                  f"{ref_ms * 1e6 / T:9.1f}   -> {ref_ms / ms:5.2f}x")
    if T == 8192:
        print(f"  {'chunked C=16 [earlier]':<22} {12.193:8.3f} "
              f"{12.193 * 1e6 / T:9.1f}   -> {12.193 / ms:5.2f}x")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--chunk", type=int, default=16, help="tokens per prep program")
    p.add_argument("--bv", type=int, default=16, help="value split of the recurrence")
    p.add_argument("--scale", type=float, default=128 ** -0.5)
    p.add_argument("--lower-bound", type=float, default=-5.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--tol", type=float, default=2e-2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    args = p.parse_args()

    if not args.selftest and not args.bench:
        p.print_help()
        return
    if not _HAS_NPU:
        raise SystemExit("this needs an NPU (torch_npu)")
    torch.npu.set_device(args.device)
    if args.bench:
        raise SystemExit(_bench(args))
    raise SystemExit(_selftest(args))


if __name__ == "__main__":
    main()
