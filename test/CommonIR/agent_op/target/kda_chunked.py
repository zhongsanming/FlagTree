"""Chunked KDA recurrence in Triton — first cut: correctness before speed.

Scope of this file, deliberately narrow so it can be validated on its own:

* prefill, one packed sequence per launch, one program per (sequence, head,
  value-tile).  BV is the value split: BV=32 gives 4 heads x 4 tiles = 16
  blocks at H=4, and keeps the [BV,K] state tile inside the UB budget.
* inputs are *prepared* (qn, kn, decay=exp(gate), beta).  Folding the prep back
  in is the next step; keeping it out makes this file comparable to
  chunked_ref.py line for line.
* no tracking / checkpoints, and no varlen beyond the per-sequence loop.

Why chunk = 16, and why it is safe without knowing the gate distribution
-----------------------------------------------------------------------
The intra-chunk matrix needs the relative decay

    A[t,s] = sum_j k_s[j] exp(gc_t[j] - gc_s[j]) k_t[j]        (s <= t)

and the only kernel-expressible factorisation is

    A^T = (kn / lam) @ (kn * lam)^T ,   lam = exp(gc)

so the transient factor is exp(-gc), whose magnitude is exp(|g|*C).  KDA's gate
is lower_bound*sigmoid(.), and lower_bound < 0 with |lower_bound| = 5 in every
caller here, so |g| <= 5 always and exp(-gc) <= exp(5C).  fp32 tops out at
3.4e38 = exp(88.7), so

    C <= 17 is safe for *any* gate distribution;  C = 16 leaves 3.8 decades.

Measured on both regimes (bench-like and gate_gain=-20, which saturates |g|=5),
C=16 fp32 gives A relative error 3e-7..4e-7 on the half that the algorithm
reads, and C=32 overflows to inf.  Slow decay would permit larger C, but 16 is
the largest value that is safe unconditionally, so it is the one to build on.

Known-slow spots, in the order I would attack them
--------------------------------------------------
1. the whole thing is fp32; the four [C,K]x[K,V] matmuls should be bf16-in /
   fp32-accumulate on the cube (probe: 7.4 TFLOP/s single core).  The state
   itself must stay fp32 in UB because its increments accumulate across chunks.
2. `tl.trans` on the [C,K] and [C,C] tiles: cheap here, but they decide whether
   the cube sees a natural layout.
3. the triangular solve is C sequential masked rank-1 updates.  At C=16 that is
   16 dependent vector steps per chunk; a (I-U)(I+U^2)(I+U^4)(I+U^8) expansion
   would cut it to ~5 but needs more matmuls.
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


@triton.jit
def _chunked_kernel(
    QN, KN, VV, DECAY, BETA, OUT, STATE, STARTS, INDICES,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
    BV: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.program_id(1)
    iv = tl.program_id(2)

    bos = tl.load(STARTS + seq).to(tl.int64)
    eos = tl.load(STARTS + seq + 1).to(tl.int64)
    n_tok = eos - bos
    slot = tl.load(INDICES + seq).to(tl.int64)

    cidx = tl.arange(0, C)
    ks = tl.arange(0, K)
    vs = iv * BV + tl.arange(0, BV)          # this program's value rows

    # State tile: the pool is [slots,H,V,K] viewed as [slots,H,K,V], i.e. K is
    # contiguous, so BV value-rows are BV*K contiguous floats.  Kept as [BV,K]
    # (v-major) so no big transpose is ever needed: every projection below is
    # written transposed instead.
    #
    # BV is what makes this fit in UB.  At BV=V the [V,K] tile is 64 KB and it
    # is an operand of two dots, each staged into L1 and doubled by the auto
    # multibuffer -> the compiler asked for 379 KB against 192 KB available.
    # 16 or 32 per program brings that back inside the budget.
    sbase = (slot * H + h) * (K * V) + iv * (BV * K)
    if slot > 0:
        st = tl.reshape(tl.load(STATE + sbase + tl.arange(0, BV * K)), [BV, K])
    else:
        st = tl.zeros([BV, K], tl.float32)

    eye = (cidx[:, None] == cidx[None, :]).to(tl.float32)      # [C,C]
    upper = cidx[:, None] < cidx[None, :]                      # s < t

    for c0 in range(0, n_tok, C):
        rows = c0 + cidx
        live = rows < n_tok

        koff = (bos + rows)[:, None] * (H * K) + h * K + ks[None, :]
        qn = tl.load(QN + koff, mask=live[:, None], other=0.0)
        kn = tl.load(KN + koff, mask=live[:, None], other=0.0)
        # other=1.0: a padded row must not advance the cumulative gate, so that
        # gc at the tail equals gc at the last live token.
        dc = tl.load(DECAY + koff, mask=live[:, None], other=1.0)
        beta = tl.load(BETA + (bos + rows) * H + h, mask=live, other=0.0)
        voff = (bos + rows)[:, None] * (H * V) + h * V + vs[None, :]
        vv = tl.load(VV + voff, mask=live[:, None], other=0.0).to(tl.float32)
        gc = tl.cumsum(tl.log(dc), axis=0)                # [C,K], <= 0
        lam = tl.exp(gc)                                  # [C,K], <= 1
        ktil = kn * lam                                   # k~ = lam * k
        qtil = qn * lam
        krec = kn * tl.exp(-gc)                           # k / lam  (the big one)

        # A^T[s,t] and Aq^T[s,t], one matmul each instead of a [C,C,K] tensor.
        #
        # Orientation, because getting it backwards is silent: with
        #     AT[m,n] = sum_j kn[m,j] kn[n,j] exp(gc_n - gc_m),  m=s, n=t,
        # and gc decreasing in t, exp(gc_n-gc_m) > 1 exactly when t < s.  So the
        # *divergent* half is s > t and everything the algorithm reads lives in
        # s <= t.  MmT keeps s < t (its diagonal comes from `eye`); AqT needs
        # s <= t, because out_t = sum_{s<=t} Aq[t,s] u_s and the diagonal
        # (exp(0) = 1) is the token's own term.
        #
        # Masking the wrong half is not caught by the magnitude: the largest
        # term is the s = C-1 row, which survives both masks for every t, so the
        # output stays at ~1e17 either way.
        keep_lo = cidx[:, None] <= cidx[None, :]           # s <= t
        AT = tl.dot(krec, tl.trans(ktil))                  # [C,C]
        AqT = tl.where(keep_lo, tl.dot(krec, tl.trans(qtil)), 0.0)
        # Mm u = rhs has Mm[t,s] = beta_t A[t,s], unit lower triangular, so its
        # transpose is unit *upper* in [s,t] -- which is the orientation the
        # column sweep below consumes.
        MmT = eye + tl.where(upper, AT * beta[None, :], 0.0)

        # rhs = beta * (v - k~ St^T), transposed to [BV,C] so `st` stays [BV,K].
        rhsT = beta[None, :] * (tl.trans(vv) - tl.dot(st, tl.trans(ktil)))

        # uT MmT = rhsT, solved as a column sweep: after step t, column t is
        # final and is subtracted from every later column.
        uT = rhsT
        for t in tl.static_range(C):
            col = tl.sum(tl.where(cidx[None, :] == t, uT, 0.0), axis=1)   # [BV]
            row = tl.sum(tl.where(cidx[:, None] == t, MmT, 0.0), axis=0)  # [C]
            coef = tl.where(cidx > t, row, 0.0)                          # s > t
            uT = uT - col[:, None] * coef[None, :]

        # out^T = St q~^T + uT Aq^T, taken before the state update.  AqT is
        # already Aq transposed by construction (AqT[s,t] = Aq[t,s]), so it is
        # used directly -- transposing it again would contract the wrong axis.
        outT = tl.dot(st, tl.trans(qtil)) + tl.dot(uT, AqT)

        # St = lam_last * St + uT @ kbar, with kbar_c = k_c exp(gc_last - gc_c).
        gc_last = tl.sum(tl.where(cidx[:, None] == C - 1, gc, 0.0), axis=0)
        kbar = kn * tl.exp(gc_last[None, :] - gc)
        st = tl.exp(gc_last)[None, :] * st + tl.dot(uT, kbar)

        ooff = ((bos + rows)[:, None] * (H * V) + h * V
                + (iv * BV + tl.arange(0, BV))[None, :])
        # Store in token-major [C,BV] order, i.e. the order OUT is laid out in.
        # Handing the store a [BV,C] tile makes the compiler insert a
        # transposing cbuf->cbuf copy, which this backend rejects outright
        # ("Unsupported copy from cbuf to cbuf"); an explicit tl.trans is a UB
        # transpose and is supported.
        tl.store(OUT + ooff, tl.trans(outT).to(OUT.dtype.element_ty),
                 mask=live[:, None])

    if slot > 0:
        tl.store(STATE + sbase + tl.arange(0, BV * K),
                 tl.reshape(st, [BV * K]))


# ---------------------------------------------------------------------------
# Host wrapper: prepared inputs, one program per (sequence, head, value-tile)
# ---------------------------------------------------------------------------
def run_chunked(qn: torch.Tensor, kn: torch.Tensor, v: torch.Tensor,
                decay: torch.Tensor, beta: torch.Tensor, state: torch.Tensor,
                indices: torch.Tensor, cu_seqlens: torch.Tensor,
                C: int = 16, BV: int = 32) -> torch.Tensor:
    """qn, kn, decay: [T,H,K] fp32; v: [T,H,V]; beta: [T,H]; state: [slots,H,K,V].

    `state` is mutated in place.  Returns out shaped like `v`.
    Only the layout the kernels in this repo require is accepted: the pool must
    be K-contiguous (stride(2)==1) and dense in slot/head.

    BV is the value split: grid = (sequences, H, V//BV).  It is a UB budget
    knob first (the [BV,K] state tile is staged into L1 for every dot and the
    auto multibuffer doubles that) and an occupancy knob second.  32 keeps
    4 heads x 4 tiles = 16 blocks busy for H=4.
    """
    T, H, K = qn.shape
    V = v.shape[-1]
    assert state.stride(2) == 1 and state.stride(3) == K, "state must be K-contiguous"
    assert state.stride(1) == K * V and state.stride(0) == H * K * V, "dense pool"
    if V % BV:
        raise ValueError(f"BV={BV} must divide V={V}")
    out = torch.empty_like(v)
    grid = (indices.numel(), H, V // BV)
    _chunked_kernel[grid](
        qn, kn, v, decay, beta, out, state, cu_seqlens, indices,
        H=H, K=K, V=V, C=C, BV=BV,
        num_warps=1, num_stages=1, multibuffer=False,
    )
    return out


# ---------------------------------------------------------------------------
# Self-test against the serial reference
# ---------------------------------------------------------------------------
def _selftest(args) -> int:
    from chunked_ref import make_inputs, serial_ref

    dev = args.device
    ok = True
    # A second length that is not a multiple of C, so the padded tail of the last
    # chunk is exercised: padded rows carry decay=1 (the cumulative gate stops)
    # and kn=0, which is what makes gc[-1] equal gc at the last live token.
    lengths = (args.tokens, max(args.chunk, args.tokens - 6))
    for tok in lengths:
        for gain in (1.0, -20.0):
            # Reference in fp64 on the CPU; the kernel gets the same values in fp32
            # on the device, so the comparison is "kernel error", not "input error".
            base = make_inputs(tok, args.heads, args.dim, args.seed,
                               torch.float64, gate_gain=gain)
            ref_out, ref_S = serial_ref(base["qn"], base["kn"], base["v"],
                                        base["decay"], base["beta"], base["S0"])
            inp = {k: t.float().to(dev) for k, t in base.items()}

            H, K, V = inp["S0"].shape
            # [slots,H,V,K] contiguous then viewed as [slots,H,K,V]: that is the
            # only layout the kernel accepts (stride(2)==1).
            pool = torch.zeros(2, H, V, K, dtype=torch.float32, device=dev)
            pool[1] = inp["S0"].transpose(-1, -2)
            state = pool.transpose(-1, -2)

            out = run_chunked(
                inp["qn"], inp["kn"], inp["v"], inp["decay"], inp["beta"], state,
                torch.tensor([1], dtype=torch.int32, device=dev),
                torch.tensor([0, inp["qn"].shape[0]], dtype=torch.int32, device=dev),
                C=args.chunk,
                BV=args.bv,
            )
            torch.npu.synchronize()

            out_err = (out.float().cpu().double() - ref_out).abs().max().item()
            state_err = (state[1].float().cpu().double() - ref_S).abs().max().item()
            out_rel = out_err / ref_out.abs().max().item()
            state_rel = state_err / ref_S.abs().max().item()
            # A magnitude ratio separates the two failure modes we have actually
            # hit: wildly wrong values (~1e15) mean a divergent half was summed,
            # while ~2 means the right values went to the wrong places.
            mag = out.float().abs().max().item() / ref_out.abs().max().item()
            good = out_rel < args.tol and state_rel < args.tol
            ok = ok and good
            print(f"T={tok:<6d} gate_gain={gain:6.1f}  out rel {out_rel:.3e}  "
                  f"state rel {state_rel:.3e}  |out| ratio {mag:9.2e}  "
                  f"{'PASS' if good else 'FAIL'}")

    print(f"\nchunk={args.chunk} bv={args.bv} tokens={args.tokens} "
          f"heads={args.heads} dim={args.dim}  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _clone_kcontig(state: torch.Tensor) -> torch.Tensor:
    """Clone a state pool while preserving the [V,K] backing.

    A plain ``.clone()`` keeps the logical [slots,H,K,V] shape but gives it
    contiguous strides for *that* shape, i.e. stride(2)==V instead of 1 -- the
    transposed layout the kernel rejects (and which, if it were accepted, would
    silently read the state transposed). Clone the [slots,H,V,K] buffer and
    re-view instead. The same trap is documented in bench_kda_tle.clone_state.
    """
    return state.transpose(-1, -2).clone().transpose(-1, -2)


def _bench(args) -> int:
    """Time the chunked recurrence on prepared inputs.

    The kernel currently covers only the recurrence, so the honest comparisons
    are (a) the full serial kernel measured earlier on this machine and (b) the
    serial recurrence with the prep ablated, also measured earlier. Both are
    printed as context, clearly labelled as earlier measurements rather than
    re-run here.
    """
    from chunked_ref import make_inputs

    dev = args.device
    H, K = args.heads, args.dim
    base = make_inputs(args.tokens, H, K, args.seed, torch.float32)
    inp = {k: t.to(dev) for k, t in base.items()}
    # [slots,H,V,K] contiguous, viewed as [slots,H,K,V]: same construction as
    # the self-test, K-contiguous is what the kernel indexes against.
    pool = torch.zeros(2, H, K, K, dtype=torch.float32, device=dev)
    pool[1] = inp["S0"].transpose(-1, -2)
    state0 = pool.transpose(-1, -2)
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, args.tokens], dtype=torch.int32, device=dev)

    def run(st):
        return run_chunked(inp["qn"], inp["kn"], inp["v"], inp["decay"],
                           inp["beta"], st, idx, qsl, C=args.chunk, BV=args.bv)

    for _ in range(args.warmup):
        run(_clone_kcontig(state0))
    torch.npu.synchronize()
    ts = []
    for _ in range(args.reps):
        st = _clone_kcontig(state0)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        run(st)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ms = statistics.median(ts)
    us_per_tok = ms * 1e3 / args.tokens
    blocks = H * (K // args.bv)

    print(f"T={args.tokens} H={H} C={args.chunk} BV={args.bv} "
          f"blocks={blocks} reps={args.reps}")
    print(f"  chunked recurrence   {ms:9.3f} ms   {us_per_tok:8.1f} ns/token")
    # Earlier measurements on this same card, for scale only.
    for serial_tok, full_ms, abl_ms in ((8192, 8.86, 5.01), (16384, 17.6, 10.0)):
        if serial_tok == args.tokens:
            print(f"  serial, full (T={serial_tok})    {full_ms:9.3f} ms   "
                  f"{full_ms * 1e3 / serial_tok:8.1f} ns/token   "
                  f"-> {full_ms / ms:5.1f}x")
            print(f"  serial, prep-ablated       {abl_ms:9.3f} ms   "
                  f"{abl_ms * 1e3 / serial_tok:8.1f} ns/token   "
                  f"-> {abl_ms / ms:5.1f}x")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true",
                   help="compare the kernel against chunked_ref.serial_ref")
    p.add_argument("--bench", action="store_true",
                   help="time the chunked recurrence on prepared inputs")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--bv", type=int, default=32,
                   help="value split; 32 or 16 keeps the UB budget; 128 only "
                        "if the multibuffer is disabled")
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
        raise SystemExit("the self-test needs an NPU (torch_npu)")
    torch.npu.set_device(args.device)
    if args.bench:
        raise SystemExit(_bench(args))
    raise SystemExit(_selftest(args))


if __name__ == "__main__":
    main()
