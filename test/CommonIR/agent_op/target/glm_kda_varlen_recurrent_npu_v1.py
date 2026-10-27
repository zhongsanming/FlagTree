"""KDA varlen recurrence — Triton v1 (candidate improvements over v0).

Drop-in replacement for
``glm_kda_varlen_recurrent_npu_v0.py`` (same directory)
that keeps v0's validation, dispatch and generic fallback untouched and only
replaces the *optimized prefill kernel*.  Everything else is delegated to v0,
so the two arms differ in exactly one kernel (same property the repo's
``bench_kda_varlen_recurrent.py`` relies on).

What is different from v0's ``_opt_recurrent_kernel``
------------------------------------------------------
All four changes are opt-in/opt-out through environment variables, because the
target is a single 910B card (40 AIV) and none of the wins below may be assumed
without measuring.  See ``../KDA_ASCENDC_VS_TLE.md`` for the reasoning.

* ``KDA_BV`` / ``KDA_CORES`` (P2) — grid is sized for one wave instead of
  autotuning blind.  The tile count per head is the largest power of two that
  fits ``cores // H`` and ``BV = 128 // tiles``: H=4/40 cores -> BV=16
  (32 blocks), H=8/40 cores -> BV=32 (32 blocks), H=4/8 cores -> BV=64.
  Autotuning is gone (``num_warps`` has no meaning on Ascend, and BV=128 was
  still being attempted even where it cannot compile).
* ``KDA_PREFETCH`` (P1) — manual one-iteration-ahead prefetch of the per-token
  q/k/a/v rows, index-clamped so the loop stays branch-free.  This is the one
  AscendC optimisation that transfers: the hand-written kernel explicitly
  prefetches token ``i+1`` into ``buf[nxt]`` and calls it out as the reason the
  double-buffered pipeline stays full.
* ``KDA_FLAT_STATE`` (P3) — with K==V==128 the ``[BV, D]`` state tile is one
  contiguous span (``BV*D`` floats), so it can be loaded/stored with a 1-D
  index instead of a masked 2-D gather.  Default off: v0 warns that unmasked
  loads miscompile at BV>=64, and this needs checking on the target toolchain.
* ``KDA_STAGES`` / ``KDA_MULTIBUFFER`` (P1a) — pass through to the backend's
  loop pipeliner, to A/B it against the manual prefetch.

Independent of the switches, the elementwise math is folded for the
non-bit-exact contract: ``q / sqrt(s) * scale`` becomes ``q * (scale / sqrt(s))``
and ``k / sqrt(s)`` becomes ``k * (1 / sqrt(s))``, turning two 128-lane vector
divides per token into scalar reciprocals plus one multiply.  AscendC pays a
Vector->Scalar round trip per token to keep the upstream operation order; with
bit-exactness dropped, that is pure overhead.

Usage
-----
    import glm_kda_varlen_recurrent_npu_v1 as v1
    out = v1.call_v1(**kwargs)     # same kwargs as v0's public function

``call_v1`` swaps v0's ``_run_optimized`` for the duration of the call and
restores it afterwards, so a single process can benchmark both arms.
"""

from __future__ import annotations

import os
import sys

import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# v0: validation, dispatch and the generic fallback stay there
# ---------------------------------------------------------------------------
# v0 sits next to this file in the target directory; GLM53_KDA_V0_DIR only
# exists so the harness can point at another checkout (e.g. flagtree-ops).
_V0_DIR = os.environ.get(
    "GLM53_KDA_V0_DIR", os.path.dirname(os.path.abspath(__file__))
)
if _V0_DIR not in sys.path:
    sys.path.insert(0, _V0_DIR)

try:
    import glm_kda_varlen_recurrent_npu_v0 as v0
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "cannot import glm_kda_varlen_recurrent_npu_v0; point GLM53_KDA_V0_DIR "
        f"at the directory containing it (tried {_V0_DIR!r})"
    ) from exc

D = 128


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _pow2_floor(x: int) -> int:
    return 1 << (x.bit_length() - 1) if x > 0 else 1


def pick_bv(heads: int, cores: int, max_bv: int = 128) -> int:
    """Largest BV whose grid still fills one wave of `cores` vector cores.

    A work item is a (head, value-tile) pair, so a head is covered by
    ``D // BV`` blocks and the grid is ``heads * (D // BV) * num_seqs``.
    Picking the *tile count* first (floor to a power of two) and deriving BV
    from it is what keeps the grid inside one wave: flooring BV instead would
    round the tile count up and spill into a second wave.

        H=4, 40 cores -> 8 tiles -> BV=16 -> 32 blocks (one wave, 80 % full)
        H=8, 40 cores -> 4 tiles -> BV=32 -> 32 blocks (one wave, 80 % full)
        H=4,  8 cores -> 2 tiles -> BV=64 ->  8 blocks
        H=8,  8 cores -> 1 tile  -> BV=128 -> 8 blocks
    """
    tiles = _pow2_floor(max(1, cores // heads))
    bv = D // tiles
    return max(16, min(max_bv, bv))


# ---------------------------------------------------------------------------
# The optimized prefill kernel
# ---------------------------------------------------------------------------
@triton.jit
def _opt_recurrent_kernel_v1(
    Q, K, V, A, B, ALOG, BIAS,
    OUT, STATE, INDICES, STARTS,
    TRACK_INDICES, TRACK_LENS,
    scale, lower_bound,
    TRACK_STATE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    FLAT_STATE: tl.constexpr,
    PREFETCH: tl.constexpr,
    SKIP_NORMS: tl.constexpr,
    SKIP_GATE: tl.constexpr,
    SKIP_VRED: tl.constexpr,
    SKIP_OUTRED: tl.constexpr,
    SKIP_LOAD: tl.constexpr,
    SCHED_OVERLAP: tl.constexpr,
    SCHED_FUSED: tl.constexpr,
):
    """One program per (value tile, sequence, head); state tile held in UB.

    Same math as v0, with the per-token inputs optionally fetched one
    iteration ahead and the normalization reciprocals hoisted onto the scalar
    path (``q * (scale / sqrt(s))`` instead of ``q / sqrt(s) * scale``).
    """
    iv = tl.program_id(0)
    i_n = tl.program_id(1)
    head = tl.program_id(2)

    bos = tl.load(STARTS + i_n).to(tl.int64)
    eos = tl.load(STARTS + i_n + 1).to(tl.int64)
    seq_len = eos - bos

    ks = tl.arange(0, D)
    vs = iv * BV + tl.arange(0, BV)
    # All-true for K==V==128 and BV | 128, but v0 found that unmasked loads
    # miscompile on this backend at BV>=64, so they are kept.
    mask_k = ks < D
    mask_v = vs < D
    mask_h = mask_v[:, None] & mask_k[None, :]

    # Per-head constants: loaded once, reused for every token.
    b_alog = tl.load(ALOG + head).to(tl.float32)
    b_bias = tl.load(BIAS + head * D + ks, mask=mask_k, other=0.0).to(tl.float32)
    decay_base = tl.exp(b_alog)

    # int64 for the state offset: `slot * S0` overflows int32 on a large pool
    # (S0 = H*K*V is already 1<<20 at H=64), which v0's int32 path does not
    # guard against.
    slot = tl.load(INDICES + i_n).to(tl.int64)

    if FLAT_STATE:
        # K-contiguous pool: a (header, value-tile) block is BV*D contiguous
        # floats starting at iv*BV*D, so no 2-D mask and no gather.
        pflat = (slot * S0 + head * S1 + iv * (BV * D)
                 + tl.arange(0, BV * D))
        state = tl.zeros([BV, D], tl.float32)
        if slot > 0:
            state = tl.reshape(tl.load(STATE + pflat), [BV, D])
    else:
        pstate = (STATE + slot * S0 + head * S1
                  + ks[None, :] * SK + vs[:, None] * SV)
        state = tl.zeros([BV, D], tl.float32)
        if slot > 0:
            state = tl.load(pstate, mask=mask_h, other=0.0).to(tl.float32)

    if TRACK_STATE:
        track_slot = tl.load(TRACK_INDICES + i_n).to(tl.int64)
        track_len = tl.load(TRACK_LENS + i_n)
        ptrack = (STATE + track_slot * S0 + head * S1
                  + ks[None, :] * SK + vs[:, None] * SV)
        if track_slot > 0 and track_len == 0:
            tl.store(ptrack, state, mask=mask_h)

    # Hoisted pointer bases: only the token offset varies in the loop.
    q_base = Q + head * D + ks
    k_base = K + head * D + ks
    v_base = V + head * D + vs
    a_base = A + head * D + ks
    b_base = B + head
    o_base = OUT + head * D + vs
    stride_t = H * D
    stride_t_b = H

    if PREFETCH:
        # Registers holding token i's inputs while token i+1's DMA is in
        # flight.  The index is clamped rather than branched so a short or
        # empty sequence cannot read out of bounds.
        p0 = tl.maximum(tl.minimum(bos, eos - 1), 0)
        t0 = p0 * stride_t
        qn = tl.load(q_base + t0, mask=mask_k, other=0.0).to(tl.float32)
        kn = tl.load(k_base + t0, mask=mask_k, other=0.0).to(tl.float32)
        vn = tl.load(v_base + t0, mask=mask_v, other=0.0).to(tl.float32)
        an = tl.load(a_base + t0, mask=mask_k, other=0.0).to(tl.float32)
        bn = tl.load(b_base + p0 * stride_t_b).to(tl.float32)

    for i in range(seq_len):
        pcur = bos + i

        if PREFETCH:
            q = qn
            k = kn
            v = vn
            a = an
            b = bn
            pnxt = bos + tl.minimum(i + 1, seq_len - 1)
            tn = pnxt * stride_t
            qn = tl.load(q_base + tn, mask=mask_k, other=0.0).to(tl.float32)
            kn = tl.load(k_base + tn, mask=mask_k, other=0.0).to(tl.float32)
            vn = tl.load(v_base + tn, mask=mask_v, other=0.0).to(tl.float32)
            an = tl.load(a_base + tn, mask=mask_k, other=0.0).to(tl.float32)
            bn = tl.load(b_base + pnxt * stride_t_b).to(tl.float32)
        else:
            tcur = pcur * stride_t
            if SKIP_LOAD:
                # Timing probe only: read token 0 of the sequence for every
                # token, so the DMA is loop-invariant and cache-resident.  The
                # difference against the control is how much of the per-token
                # cost is the load dependency rather than the compute chain.
                tcur = bos * stride_t
            q = tl.load(q_base + tcur, mask=mask_k, other=0.0).to(tl.float32)
            k = tl.load(k_base + tcur, mask=mask_k, other=0.0).to(tl.float32)
            v = tl.load(v_base + tcur, mask=mask_v, other=0.0).to(tl.float32)
            a = tl.load(a_base + tcur, mask=mask_k, other=0.0).to(tl.float32)
            b = tl.load(b_base + pcur * stride_t_b).to(tl.float32)

        # --- ablations -----------------------------------------------------
        # Timing probes, not correct math.  The measured per-token budget is an
        # affine fit: ~0.79 us of BV-independent time plus ~0.018 us per BV
        # unit, i.e. 74 % of the time at BV=16 is a serial chain that does not
        # scale with the tile.  These switches attribute that chain: the two
        # cross-lane reductions, the gate/beta exponentials, the two state
        # reductions, or the load dependency.
        if SKIP_NORMS:
            q = q * scale
        else:
            s_q = tl.sum(q * q, axis=0)
            s_k = tl.sum(k * k, axis=0)
            q = q * (scale / tl.sqrt(s_q + 1e-6))
            k = k * (1.0 / tl.sqrt(s_k + 1e-6))

        if SKIP_GATE:
            decay = tl.full([D], 1.0, tl.float32)
            beta = 0.5
        else:
            x = a + b_bias
            gate = lower_bound / (1.0 + tl.exp(-(decay_base * x)))
            decay = tl.exp(gate)
            beta = 1.0 / (1.0 + tl.exp(-b))

        # ── state update ────────────────────────────────────────────────
        # Baseline (the shape upstream uses):
        #     state *= decay; v = beta*(v - sum(state*k)); state += k v^T
        #     out = sum(state*q)
        # `out` therefore sits downstream of the outer-product update, so the
        # two reductions cannot overlap and the loop pays both latencies back
        # to back.  The measurements (see KDA_ASCENDC_VS_TLE.md 6) put those
        # two stages at ~49 % of the wall clock.
        #
        # SCHED_OVERLAP/SCHED_FUSED use the identity
        #     out = sum(state_d*q) + (k.q)*v,   state_d = state*decay
        # so both reductions depend only on state_d, become independent, and
        # `out` no longer waits for the update.  This is a reassociation, legal
        # because the contract is tolerance-based, not bit-exact.  The probes
        # assume the baseline order, so the launcher makes the two mutually
        # exclusive.
        if SCHED_OVERLAP or SCHED_FUSED:
            state_d = state * decay[None, :]
            if SCHED_FUSED:
                # One reduce, two results: reduce the interleaved [BV, K, 2]
                # tile over K, then split the trailing axis (which is 2).
                pair = tl.sum(
                    tl.join(state_d * k[None, :], state_d * q[None, :]),
                    axis=1,
                )
                v_corr, out_part = tl.split(pair)
            else:
                v_corr = tl.sum(state_d * k[None, :], axis=1)
                out_part = tl.sum(state_d * q[None, :], axis=1)
            kq = tl.sum(k * q, axis=0)

            v = (v - v_corr) * beta
            state = state_d + k[None, :] * v[:, None]

            if TRACK_STATE:
                if track_slot > 0 and i + 1 == track_len:
                    tl.store(ptrack, state, mask=mask_h)

            out = out_part + kq * v
        else:
            state = state * decay[None, :]
            if SKIP_VRED:
                v = v * beta
            else:
                v = v - tl.sum(state * k[None, :], axis=1)
                v = v * beta
            state = state + k[None, :] * v[:, None]

            if TRACK_STATE:
                if track_slot > 0 and i + 1 == track_len:
                    tl.store(ptrack, state, mask=mask_h)

            if SKIP_OUTRED:
                out = v * 0.5
            else:
                out = tl.sum(state * q[None, :], axis=1)

        tl.store(o_base + pcur * stride_t,
                 out.to(OUT.dtype.element_ty), mask=mask_v)

    if slot > 0:
        if FLAT_STATE:
            pflat_out = (slot * S0 + head * S1 + iv * (BV * D)
                         + tl.arange(0, BV * D))
            tl.store(STATE + pflat_out, tl.reshape(state, [BV * D]))
        else:
            tl.store(pstate, state, mask=mask_h)


# ---------------------------------------------------------------------------
# Launcher: same keyword contract as v0._run_optimized
# ---------------------------------------------------------------------------
_ABLATIONS = ("norms", "gate", "vred", "outred", "load")


def _ablation_flags() -> dict:
    """Parse KDA_ABLATE (comma list, or 'all') into constexpr flags.

    Probes are timing-only: they deliberately compute the wrong answer, so they
    must never be used outside a bench run.  'all' keeps the loads and drops
    every other per-token stage, which is the floor the serial chain costs.
    """
    spec = os.environ.get("KDA_ABLATE", "").strip()
    if not spec:
        names = set()
    elif spec in ("all", "*"):
        names = set(_ABLATIONS) - {"load"}
    else:
        names = {part.strip() for part in spec.split(",") if part.strip()}
        unknown = names - set(_ABLATIONS)
        if unknown:
            raise ValueError(
                f"KDA_ABLATE has unknown entries {sorted(unknown)}; "
                f"known: {sorted(_ABLATIONS)} or 'all'"
            )
    return {f"SKIP_{name.upper()}": (name in names) for name in _ABLATIONS}


_SCHED_MODES = {"base": (False, False), "overlap": (True, False), "fused": (True, True)}


def _sched_flags() -> tuple[bool, bool]:
    """Parse KDA_SCHED into (SCHED_OVERLAP, SCHED_FUSED).

    base     upstream shape: out = sum(state_new*q), i.e. downstream of the
             outer-product update and unable to overlap with the v reduction.
    overlap  out = sum(state_d*q) + (k.q)*v, both reductions independent.
    fused    same, with the two reductions emitted as one interleaved reduce.
    """
    spec = os.environ.get("KDA_SCHED", "base").strip().lower() or "base"
    if spec not in _SCHED_MODES:
        raise ValueError(
            f"KDA_SCHED must be one of {sorted(_SCHED_MODES)}, got {spec!r}"
        )
    return _SCHED_MODES[spec]


def _run_optimized_v1(
    *, q, k, v, a, b,
    A_log, dt_bias, output, state, indices, starts,
    scale, lower_bound,
    track_indices, track_lens,
    key_dim, num_seqs,
):
    heads = q.shape[1]
    track = track_indices is not None

    cores = _env_int("KDA_CORES", 40)
    bv = _env_int("KDA_BV", 0)
    if bv <= 0:
        bv = pick_bv(heads, cores, _env_int("KDA_MAX_BV", 128))
    if bv < 16 or bv > 128 or (128 % bv) != 0:
        raise ValueError(f"KDA_BV must divide 128 and be in [16, 128], got {bv}")

    # The flat path also needs slot/head to be dense: stride(1) == K*V and
    # stride(0) == H*stride(1), which is what the production pool is.
    flat = (
        _env_int("KDA_FLAT_STATE", 0) != 0
        and state.stride(2) == 1
        and state.stride(3) == key_dim
        and state.stride(1) == key_dim * key_dim
        and state.stride(0) == heads * key_dim * key_dim
    )

    ablate = _ablation_flags()
    prefetch = _env_int("KDA_PREFETCH", 0) != 0
    if prefetch and ablate["SKIP_LOAD"]:
        raise ValueError("KDA_ABLATE=load needs KDA_PREFETCH=0")
    sched_overlap, sched_fused = _sched_flags()
    if (sched_overlap or sched_fused) and any(ablate.values()):
        # The probes assume the baseline order, so keeping both knobs live
        # would make the attribution table meaningless.
        raise ValueError("KDA_SCHED and KDA_ABLATE are mutually exclusive")

    grid = (key_dim // bv, num_seqs, heads)
    _opt_recurrent_kernel_v1[grid](
        q, k, v, a, b,
        A_log, dt_bias,
        output, state, indices, starts,
        track_indices if track else indices,
        track_lens if track else starts,
        scale, lower_bound,
        TRACK_STATE=track,
        H=heads, D=key_dim, BV=bv,
        S0=state.stride(0), S1=state.stride(1),
        SK=state.stride(2), SV=state.stride(3),
        FLAT_STATE=flat,
        PREFETCH=prefetch,
        SKIP_NORMS=ablate["SKIP_NORMS"],
        SKIP_GATE=ablate["SKIP_GATE"],
        SKIP_VRED=ablate["SKIP_VRED"],
        SKIP_OUTRED=ablate["SKIP_OUTRED"],
        SKIP_LOAD=ablate["SKIP_LOAD"],
        SCHED_OVERLAP=sched_overlap,
        SCHED_FUSED=sched_fused,
        num_warps=_env_int("KDA_NUM_WARPS", 1),
        num_stages=_env_int("KDA_STAGES", 1),
        multibuffer=_env_int("KDA_MULTIBUFFER", 0) != 0,
    )


# ---------------------------------------------------------------------------
# Public entry: v0's contract, v1's kernel
# ---------------------------------------------------------------------------
def call_v1(**kwargs):
    """``glm_kda_varlen_recurrent_npu(**kwargs)`` with v1's optimized kernel.

    v0 owns the validation, the dispatch and the generic fallback; only its
    ``_run_optimized`` is swapped, and swapped back, so the process can go on
    to benchmark v0 in the same run.
    """
    original = v0._run_optimized
    v0._run_optimized = _run_optimized_v1
    try:
        return v0.glm_kda_varlen_recurrent_npu(**kwargs)
    finally:
        v0._run_optimized = original


def call_v0(**kwargs):
    return v0.glm_kda_varlen_recurrent_npu(**kwargs)


def config_summary() -> dict:
    return {
        "cores": _env_int("KDA_CORES", 40),
        "bv": _env_int("KDA_BV", 0) or None,
        "prefetch": _env_int("KDA_PREFETCH", 0) != 0,
        "flat_state": _env_int("KDA_FLAT_STATE", 0) != 0,
        "stages": _env_int("KDA_STAGES", 1),
        "multibuffer": _env_int("KDA_MULTIBUFFER", 0) != 0,
        "num_warps": _env_int("KDA_NUM_WARPS", 1),
        "ablate": os.environ.get("KDA_ABLATE", "") or None,
        "sched": os.environ.get("KDA_SCHED", "base"),
    }
