#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""commonir_bench.py -- accuracy / performance / IR harness for the six
CommonIR(TLE-DSA) Ascend kernels in ``test/CommonIR/Ascend/success_case/``
versus their non-CommonIR counterparts in FlagGems / FlagGems-vllm.

Why a driver + per-impl subprocess
----------------------------------
Every measurement runs in its own subprocess because

* importing ``flaggems_vllm`` and/or ``flag_gems`` monkey-patches ``torch`` and
  registers custom ops -- two copies in one process silently shadow each other;
* these NPU kernels can fault the MTE engine or hang, and a wedged NPU process
  is not recoverable;
* ``TRITON_DUMP_DIR`` / ``MLIR_ENABLE_DUMP`` are process-wide environment
  variables, so per-impl IR dumps need per-impl processes.

Stage layout (all stages are independently re-runnable)
-------------------------------------------------------
    produce   one worker subprocess per (op, case, impl) -> timings + artifacts
    compare   CPU-only; loads both artifacts of a case and computes the diff
    report    aggregates everything into json/csv/md

Because ``compare`` never touches the NPU it can be re-run on this machine, on
the copied-back output folder, without the card.

Usage
-----
    # full pipeline, quick case set, with IR dumps
    python commonir_bench.py run --outdir ./out/run1 --cases quick --dump-ir

    # specific ops / cases
    python commonir_bench.py run --outdir ./out/run1 --ops compressor,gqa \\
        --cases prefill_small,b2_s256x512_h32x8_d128_causal

    # re-compare an existing output folder on a CPU-only box
    python commonir_bench.py compare --outdir ./out/run1
    python commonir_bench.py report  --outdir ./out/run1

    # single worker (used internally; handy for debugging)
    python commonir_bench.py worker --outdir ./out/run1 --op gqa \\
        --case b1_s128_h16x8_d128_causal --impl new

See README.md for the op/case registry and for the counterpart mapping.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNEL_DIR_DEFAULT = os.path.abspath(
    os.path.join(_HERE, "..", "success_case")
)
_SELF = os.path.abspath(__file__)

NEW_IMPL = "new"  # CommonIR / TLE rewrite (test/CommonIR/Ascend/success_case)
REF_IMPL = "ref"  # non-CommonIR counterpart
IMPLS = (NEW_IMPL, REF_IMPL)

_IMPL_LABEL = {
    NEW_IMPL: "commonir (tle.dsa)",
    REF_IMPL: "counterpart (no tle.dsa)",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _log(msg):
    print(msg, flush=True)


def _dt(name):
    import torch

    return getattr(torch, name)


def _sync(device):
    import torch

    if str(device).startswith("npu") and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _device_index(device):
    parts = str(device).split(":")
    return int(parts[1]) if len(parts) > 1 else 0


def _safe(obj):
    """Make an object json-serialisable."""
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _json_dump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_safe(obj), f, indent=2, sort_keys=True)


def _json_load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _tagged(stem, tag):
    return f"{stem}@{tag}" if tag else stem


def _art_path(outdir, op, case_id, impl, tag=""):
    return os.path.join(
        outdir, "artifacts", _tagged(f"{op}__{case_id}__{impl}", tag) + ".pt"
    )


def _raw_path(outdir, op, case_id, impl, tag=""):
    return os.path.join(
        outdir, "raw", _tagged(f"{op}__{case_id}__{impl}", tag) + ".json"
    )


def _log_path(outdir, op, case_id, impl, tag=""):
    return os.path.join(
        outdir, "logs", _tagged(f"{op}__{case_id}__{impl}", tag) + ".log"
    )


# --------------------------------------------------------------------------- #
# New-kernel module loader
# --------------------------------------------------------------------------- #


_RUN_NOTES = []


def _note(msg):
    """Process-wide note list; survives an exception so patches are auditable."""
    if msg not in _RUN_NOTES:
        _RUN_NOTES.append(msg)


def _load_kernel_module(op, kernel_dir):
    """Import ``<kernel_dir>/<op>.py`` under a private module name (cached)."""
    key = (op, os.path.abspath(kernel_dir))
    cached = _KERNEL_MODULES.get(key)
    if cached is not None:
        return cached
    path = os.path.join(kernel_dir, op + ".py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"CommonIR kernel not found: {path} (use --kernel-dir)"
        )
    mod_name = f"_commonir_kernel_{op}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise AttributeError(f"{path} has no top-level run()")
    _KERNEL_MODULES[key] = mod
    return mod


_KERNEL_MODULES = {}


# =========================================================================== #
# Case registries
# =========================================================================== #
#
# `quick` is meant to finish in a couple of minutes; `default` covers every
# code path that is cheap enough to be worth having; `full` additionally
# includes the heavy production shapes (multi-GB state caches / outputs).

CASES = {}


# --------------------------------------------------------------------- #
# 1. causal_conv1d_fn
# --------------------------------------------------------------------- #
CASES["causal_conv1d_fn"] = {
    "quick": (
        "b4_d64_s8_pad",
        "b4_d4096_s249",
    ),
    "default": (
        "b4_d64_s8_pad",
        "b4_d4096_s249",
        "b10_d4096_s1024_pad_fp16",
        "b2_d256_s512_nopad",
    ),
    "full": (
        "b4_d64_s8_pad",
        "b4_d4096_s249",
        "b10_d4096_s4096_pad_fp16",
        "b2_d256_s512_nopad",
        "b1_d128_s2048_pad",
        "b2_d128_s2048_avg_long",
    ),
    "heavy": (),
}
_CC1D_SPECS = {
    # id: (batch, dim, total_len, with_pad, dtype, has_bias, silu)
    "b4_d64_s8_pad": (4, 64, 8, True, "bfloat16", True, True),
    "b4_d4096_s249": (4, 4096, 249, False, "bfloat16", True, True),
    "b10_d4096_s1024_pad_fp16": (10, 4096, 1024, True, "float16", True, True),
    "b2_d256_s512_nopad": (2, 256, 512, False, "bfloat16", True, True),
    "b10_d4096_s4096_pad_fp16": (10, 4096, 4096, True, "float16", True, True),
    "b1_d128_s2048_pad": (1, 128, 2048, True, "bfloat16", False, False),
    # avg tokens/sequence >= 256 --> exercises the non-HEAD_DSA (masked) head
    "b2_d128_s2048_avg_long": (2, 128, 2048, True, "bfloat16", True, True),
}


# --------------------------------------------------------------------- #
# 2. compressor  (mirrors FlagGems-vllm tests/compressor_cases.py)
# --------------------------------------------------------------------- #
CASES["compressor"] = {
    "quick": ("prefill_small", "decode3"),
    "default": (
        "prefill_small",
        "decode3",
        "prefill0_b2_rank3",
        "prefill1_hd128",
        "decode5_c128",
    ),
    "full": (
        "prefill0",
        "prefill1_hd128",
        "prefill2_c128",
        "prefill0_b2",
        "prefill0_b2_rank3",
        "decode0_start8195",
        "decode3",
        "decode5_c128",
        "prefill_small",
    ),
    # ~2 GB fp32 state_cache apiece: opt in explicitly ("--cases heavy")
    "heavy": ("decode0_b32",),
}
_COMPRESSOR_SPECS = {
    # id: (batch, hidden, seq_spec, head_dim, block_size, rope_dim, ratio, coff, start_spec)
    "prefill0": (1, 4096, 8192, 512, 128, 64, 4, 2, 0),
    "prefill1_hd128": (1, 4096, 8192, 128, 128, 64, 4, 2, 0),
    "prefill2_c128": (1, 4096, 8192, 512, 128, 64, 128, 1, 0),
    "prefill0_b2": (2, 4096, 4096, 512, 128, 64, 4, 2, 0),
    "prefill0_b2_rank3": (2, 4096, 4096, 512, 128, 64, 4, 2, 0),
    "decode0_start8195": (1, 4096, 1, 512, 128, 64, 4, 2, 8195),
    "decode3": (8, 4096, 3, 512, 128, 64, 4, 2, 8193),
    "decode5_c128": (8, 4096, 3, 512, 128, 64, 128, 1, 8193),
    "decode0_b32": (32, 4096, 1, 512, 128, 64, 4, 2, 8195),
    "prefill_small": (1, 2048, 1024, 512, 128, 64, 4, 2, 0),
}
_COMPRESSOR_RANK3 = {"prefill0_b2_rank3"}


# --------------------------------------------------------------------- #
# 3. gqa
# --------------------------------------------------------------------- #
CASES["gqa"] = {
    "quick": ("b1_s128_h16x8_d128_causal", "b1_s1024_h8x2_d64_noncausal"),
    "default": (
        "b1_s128_h16x8_d128_causal",
        "b1_s1024_h8x2_d64_noncausal",
        "b2_s256x512_h32x8_d128_causal",
        "b4_s512_h32x4_d64_causal_fp16",
    ),
    "full": (
        "b1_s128_h16x8_d128_causal",
        "b1_s1024_h8x2_d64_noncausal",
        "b2_s256x512_h32x8_d128_causal",
        "b4_s512_h32x4_d64_causal_fp16",
        "b1_s2048_h64x8_d128_causal",
        "b1_s1x1024_h64x1_d128_decode",
    ),
}
_GQA_SPECS = {
    # id: (batch, seq_q, seq_kv, heads_q, heads_kv, head_dim, is_causal, dtype)
    "b1_s128_h16x8_d128_causal": (1, 128, 128, 16, 8, 128, True, "bfloat16"),
    "b1_s1024_h8x2_d64_noncausal": (1, 1024, 1024, 8, 2, 64, False, "bfloat16"),
    "b2_s256x512_h32x8_d128_causal": (2, 256, 512, 32, 8, 128, True, "bfloat16"),
    "b4_s512_h32x4_d64_causal_fp16": (4, 512, 512, 32, 4, 64, True, "float16"),
    "b1_s2048_h64x8_d128_causal": (1, 2048, 2048, 64, 8, 128, True, "bfloat16"),
    "b1_s1x1024_h64x1_d128_decode": (1, 1, 1024, 64, 1, 128, True, "bfloat16"),
}


# --------------------------------------------------------------------- #
# 4. kv_rmsnorm_rope_cache  (mirrors tests/test_kv_rmsnorm_rope_cache.py)
# --------------------------------------------------------------------- #
CASES["kv_rmsnorm_rope_cache"] = {
    "quick": ("decode_unaligned_fp16", "prefill_fp16"),
    "default": ("decode_unaligned_fp16", "prefill_fp16", "decode_bf16"),
    "full": (
        "decode_aligned_fp16",
        "decode_unaligned_fp16",
        "prefill_fp16",
        "decode_bf16",
    ),
}
_KV_SPECS = {
    # id: (batch, seq_len, dtype)
    "decode_aligned_fp16": (16384, 1, "float16"),
    "decode_unaligned_fp16": (3, 1, "float16"),
    "prefill_fp16": (64, 256, "float16"),
    "decode_bf16": (4096, 1, "bfloat16"),
}


# --------------------------------------------------------------------- #
# 5. lightning_indexer  (mirrors tests/lightning_indexer_utils.py)
# --------------------------------------------------------------------- #
CASES["lightning_indexer"] = {
    "quick": ("R1", "R3", "R8"),
    "default": ("R1", "R3", "R6", "R8", "R11"),
    "full": ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12"),
}
_LI_SPECS = {
    # id: (query_t, local_q_lengths, key_lengths)
    "R1": (1, (1,), (8,)),
    "R2": (16, (16,), (128,)),
    "R3": (256, (256,), (2048,)),
    "R4": (1024, (1024,), (8192,)),
    "R5": (1, (1, 0, 0, 0), (130, 0, 0, 0)),
    "R6": (1, (1, 0, 0, 0), (2050, 0, 0, 0)),
    "R7": (1, (1, 0, 0, 0), (8194, 0, 0, 0)),
    "R8": (33, (1, 16, 16), (130, 128, 128)),
    "R9": (81, (1, 0, 0, 16, 16, 16, 16, 16), (132, 130, 130, 128, 128, 128, 128, 128)),
    "R10": (2, (1, 1, 0, 0, 0, 0, 0, 0), (134, 132, 132, 130, 130, 130, 130, 130)),
    "R11": (2, (1, 1, 0, 0, 0, 0, 0, 0), (136, 136, 136, 136, 136, 0, 0, 0)),
    "R12": (1, (1, 0), (130, 2050)),
}


# --------------------------------------------------------------------- #
# 6. sparse_attn_sharedkv  (mirrors tests/sparse_attn_sharedkv_utils.py)
# --------------------------------------------------------------------- #
CASES["sparse_attn_sharedkv"] = {
    "quick": ("scfa_decode", "swa_decode", "cfa_decode"),
    "default": (
        "scfa_decode",
        "swa_decode",
        "cfa_decode",
        "swa_prefill_small",
        "cfa_prefill_small",
    ),
    "full": (
        "scfa_decode",
        "swa_decode",
        "cfa_decode",
        "scfa_prefill",
        "swa_prefill",
        "cfa_prefill",
    ),
}
_SAS_SPECS = {
    # id: (mode, q_len, kv_len, cmp_ratio, topk)
    "scfa_decode": ("SCFA", 1, 8193, 4, 512),
    "swa_decode": ("SWA", 1, 8193, 0, 0),
    "cfa_decode": ("CFA", 1, 8193, 128, 0),
    "scfa_prefill": ("SCFA", 8192, 8192, 4, 512),
    "swa_prefill": ("SWA", 8192, 8192, 0, 0),
    "cfa_prefill": ("CFA", 8192, 8192, 128, 0),
    "swa_prefill_small": ("SWA", 512, 512, 0, 0),
    "cfa_prefill_small": ("CFA", 512, 512, 128, 0),
}


OP_ORDER = (
    "causal_conv1d_fn",
    "compressor",
    "gqa",
    "kv_rmsnorm_rope_cache",
    "lightning_indexer",
    "sparse_attn_sharedkv",
)

# every op exposes the same set names so ``--cases heavy`` never KeyErrors
for _op in OP_ORDER:
    CASES[_op].setdefault("heavy", ())


# =========================================================================== #
# Input builders
# =========================================================================== #
#
# Every builder is deterministic (fixed seeds) so the `new` and `ref` workers
# reconstruct bit-identical inputs.  Each returns (kwargs, meta); `meta` holds
# everything the comparison needs but the kernel does not take.


# --------------------------------------------------------------------- #
# 1. causal_conv1d_fn
# --------------------------------------------------------------------- #
def build_causal_conv1d_fn(case_id, device):
    import numpy as np
    import torch

    batch, dim, total_len, with_pad, dtype_s, has_bias, silu = _CC1D_SPECS[case_id]
    dtype = _dt(dtype_s)
    width = 4
    pad = 3 if with_pad else 0
    padded = batch + pad

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    nsplits = padded - 1
    eos = torch.randperm(total_len - 1)[:nsplits].sort().values
    seqlens = torch.diff(
        torch.cat([torch.tensor([-1]), eos, torch.tensor([total_len - 1])])
    ).tolist()
    assert sum(seqlens) == total_len and all(s > 0 for s in seqlens)

    cumsum = torch.cumsum(torch.tensor(seqlens), dim=0).to(torch.int32)
    query_start_loc = torch.cat(
        [torch.tensor([0], dtype=torch.int32), cumsum], dim=0
    ).to(device)

    # Slice out of a wider tensor on purpose: keeps x non-contiguous in a way
    # the kernels are expected to handle (stride(dim) == 1, stride(token) != 1).
    pre = 64
    x = torch.randn(1, total_len, pre + dim, device=device, dtype=dtype)
    x = x.transpose(1, 2)[:, pre : pre + dim, :]

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None

    total_entries = batch * 10
    conv_states = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=dtype
    ).transpose(1, 2)

    has_initial_state = torch.randint(
        0, 2, (padded,), dtype=torch.bool, device=device
    )
    state_indices = torch.randperm(total_entries, dtype=torch.int32, device=device)[
        :batch
    ]
    cache_indices = torch.cat(
        [
            state_indices,
            torch.full((pad,), -1, dtype=torch.int32, device=device),
        ]
    )

    kw = dict(
        x=x.squeeze(0),
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu" if silu else None,
        pad_slot_id=-1,
    )
    meta = dict(
        seqlens=seqlens,
        live=[int(v) != -1 for v in cache_indices.tolist()],
        slots=[int(v) for v in state_indices.tolist()],
        dtype=dtype_s,
    )
    return kw, meta


# --------------------------------------------------------------------- #
# 2. compressor
# --------------------------------------------------------------------- #
def _compressor_case_layout(case_id):
    from itertools import accumulate

    spec = _COMPRESSOR_SPECS[case_id]
    batch, hidden, seq_spec, head_dim, block_size, rope_dim, ratio, coff, start_spec = spec

    def per_request(value):
        vals = (value,) * batch if isinstance(value, int) else tuple(value)
        if len(vals) != batch:
            raise ValueError(f"case {case_id}: expected {batch} values, got {vals}")
        return vals

    return dict(
        batch=batch,
        hidden=hidden,
        seq_lens=per_request(seq_spec),
        head_dim=head_dim,
        block_size=block_size,
        rope_dim=rope_dim,
        ratio=ratio,
        coff=coff,
        starts=per_request(start_spec),
        rank3=case_id in _COMPRESSOR_RANK3
        or case_id.endswith("_rank3"),
        cumulative=list(accumulate(per_request(seq_spec))),
    )


def _compressor_valid_rows(layout):
    def compressed(start, count):
        return max(
            (int(start) + max(int(count), 0)) // layout["ratio"]
            - int(start) // layout["ratio"],
            0,
        )

    return [
        compressed(start, count)
        for start, count in zip(layout["starts"], layout["seq_lens"])
    ]


def build_compressor(case_id, device):
    import torch

    lay = _compressor_case_layout(case_id)
    batch = lay["batch"]
    ratio = lay["ratio"]
    head_dim = lay["head_dim"]
    coff = lay["coff"]
    rope_dim = lay["rope_dim"]
    block_size = lay["block_size"]
    seq_lens = lay["seq_lens"]
    starts = lay["starts"]
    total_tokens = sum(seq_lens)
    proj_dim = coff * head_dim
    dtype = torch.bfloat16

    torch.manual_seed(42)

    def uniform(shape, low, high, dt):
        return (
            torch.rand(shape, dtype=torch.float32, device=device) * (high - low) + low
        ).to(dt)

    max_pos = max(s + n for s, n in zip(starts, seq_lens))
    max_blocks = (max_pos + block_size - 1) // block_size
    table = torch.arange(1, batch * max_blocks + 1, dtype=torch.int32).reshape(
        batch, max_blocks
    )
    state_blocks = batch * max_blocks + 1

    out_per_batch = (seq_lens[0] + ratio - 1) // ratio
    rope_rows = (
        batch * out_per_batch
        if lay["rank3"]
        else min(total_tokens, total_tokens // ratio + batch)
    )

    x = uniform((total_tokens, lay["hidden"]), -10, 10, dtype)
    rope_sin = uniform((rope_rows, rope_dim), -1, 1, dtype)
    rope_cos = uniform((rope_rows, rope_dim), -1, 1, dtype)
    if lay["rank3"]:
        x = x.reshape(batch, seq_lens[0], lay["hidden"])
        rope_sin = rope_sin.reshape(batch, out_per_batch, rope_dim)
        rope_cos = rope_cos.reshape(batch, out_per_batch, rope_dim)

    kw = dict(
        x=x,
        wkv=uniform((proj_dim, lay["hidden"]), -10, 10, dtype),
        wgate=uniform((proj_dim, lay["hidden"]), -10, 10, dtype),
        state_cache=uniform(
            (state_blocks, block_size, 2 * proj_dim), -10, 10, torch.float32
        ),
        ape=uniform((ratio, proj_dim), -10, 10, torch.float32),
        norm_weight=uniform((head_dim,), -10, 10, dtype),
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        state_block_table=table.to(device=device),
        cu_seqlens=None
        if lay["rank3"]
        else torch.tensor(
            [0, *lay["cumulative"][:-1], total_tokens], dtype=torch.int32
        ).to(device=device),
        seqused=None,
        start_pos=torch.tensor(starts, dtype=torch.int32, device=device),
        rope_head_dim=rope_dim,
        cmp_ratio=ratio,
        coff=coff,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=1,
    )
    meta = dict(rank3=lay["rank3"], valid_rows=_compressor_valid_rows(lay))
    return kw, meta


# --------------------------------------------------------------------- #
# 3. gqa
# --------------------------------------------------------------------- #
def build_gqa(case_id, device):
    import torch

    batch, seq_q, seq_kv, hq, hkv, hd, is_causal, dtype_s = _GQA_SPECS[case_id]
    dtype = _dt(dtype_s)
    torch.manual_seed(7)
    kw = dict(
        query=(torch.rand(batch, seq_q, hq, hd, device=device, dtype=torch.float32) * 2 - 1).to(dtype),
        key=(torch.rand(batch, seq_kv, hkv, hd, device=device, dtype=torch.float32) * 2 - 1).to(dtype),
        value=(torch.rand(batch, seq_kv, hkv, hd, device=device, dtype=torch.float32) * 2 - 1).to(dtype),
        scaleValue=-1.0,
        is_causal=is_causal,
    )
    meta = dict(dtype=dtype_s)
    return kw, meta


# --------------------------------------------------------------------- #
# 4. kv_rmsnorm_rope_cache
# --------------------------------------------------------------------- #
def build_kv_rmsnorm_rope_cache(case_id, device):
    import torch

    batch, seq_len, dtype_s = _KV_SPECS[case_id]
    dtype = _dt(dtype_s)
    rms, rope = 512, 64
    tokens = batch * seq_len
    page_size = 128
    page_num = math.ceil(tokens / page_size) + 8

    torch.manual_seed(0)
    kv = torch.randn(batch, 1, seq_len, rms + rope, dtype=dtype, device=device)
    gamma = torch.ones(rms, dtype=dtype, device=device)
    cos = torch.randn(batch, 1, seq_len, rope, dtype=dtype, device=device)
    sin = torch.randn(batch, 1, seq_len, rope, dtype=dtype, device=device)
    k_cache = torch.zeros(page_num, page_size, 1, rope, dtype=dtype, device=device)
    ckv_cache = torch.zeros(page_num, page_size, 1, rms, dtype=dtype, device=device)
    index = torch.arange(tokens, dtype=torch.int64, device=device)

    kw = dict(
        kv=kv,
        gamma=gamma,
        cos=cos,
        sin=sin,
        index=index,
        k_cache=k_cache,
        ckv_cache=ckv_cache,
        k_rope_scale=None,
        c_kv_scale=None,
        k_rope_offset=None,
        c_kv_offset=None,
        epsilon=1e-5,
        cache_mode="PA_BNSD",
        is_output_kv=True,
    )
    meta = dict(dtype=dtype_s, rms=rms, rope=rope, page_size=page_size)
    return kw, meta


# --------------------------------------------------------------------- #
# 5. lightning_indexer
# --------------------------------------------------------------------- #
_LI_QUERY_HEADS = 64
_LI_HEAD_DIM = 128
_LI_BLOCK_SIZE = 128
_LI_TOP_K = 512


def _cumulative(lengths):
    out, total = [], 0
    for n in lengths:
        total += int(n)
        out.append(total)
    return out


def build_lightning_indexer(case_id, device):
    import torch

    query_t, local_q, key_lengths = _LI_SPECS[case_id]
    torch.manual_seed(3)
    if hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed_all(3)
        except Exception:
            pass

    total_blocks = sum(math.ceil(k / _LI_BLOCK_SIZE) for k in key_lengths)
    key_blocks = total_blocks + 8
    table_width = max(2, max(math.ceil(k / _LI_BLOCK_SIZE) for k in key_lengths))

    query = torch.empty(
        (query_t, _LI_QUERY_HEADS, _LI_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-8, 8)
    key = torch.empty(
        (key_blocks, _LI_BLOCK_SIZE, 1, _LI_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-8, 8)
    weights = torch.empty(
        (query_t, _LI_QUERY_HEADS), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)

    block_table = torch.zeros(
        (len(key_lengths), table_width), dtype=torch.int32, device=device
    )
    nxt = 0
    for req, klen in enumerate(key_lengths):
        nblk = math.ceil(klen / _LI_BLOCK_SIZE)
        if nblk:
            block_table[req, :nblk] = torch.arange(
                nxt, nxt + nblk, dtype=torch.int32, device=device
            )
        nxt += nblk

    kw = dict(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=torch.tensor(
            _cumulative(local_q), dtype=torch.int32, device=device
        ),
        actual_seq_lengths_key=torch.tensor(
            list(key_lengths), dtype=torch.int32, device=device
        ),
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=_LI_TOP_K,
        sparse_mode=3,
        pre_tokens=9223372036854775807,
        next_tokens=9223372036854775807,
        return_value=False,
    )
    meta = dict(query_ends=list(_cumulative(local_q)), key_lengths=list(key_lengths))
    return kw, meta


# --------------------------------------------------------------------- #
# 6. sparse_attn_sharedkv
# --------------------------------------------------------------------- #
_SAS_Q_HEADS = 64
_SAS_KV_HEADS = 1
_SAS_HEAD_DIM = 512
_SAS_PAGE_SIZE = 128
_SAS_SOFTMAX_SCALE = 0.04419417
_SAS_SEED = 43


def _sas_paged(token_count, generator, device):
    import torch

    pages = math.ceil(token_count / _SAS_PAGE_SIZE)
    block_table = torch.randperm(
        pages, generator=generator, dtype=torch.int32, device="cpu"
    ).view(1, pages)
    kv = torch.rand(
        (pages, _SAS_PAGE_SIZE, _SAS_KV_HEADS, _SAS_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return (kv * 15 - 5).to(torch.bfloat16).to(device), block_table.to(device)


def _sas_metadata(kw, q_len, kv_len, cmp_ratio, topk, device, mode):
    """The Triton counterpart validates -- but never reads -- a 1024-i32 blob.

    Prefer the real vllm-ascend metadata op; fall back to zeros and say so.
    """
    import torch

    if mode == "zeros":
        return torch.zeros(1024, dtype=torch.int32, device=device), "zeros"
    try:
        import vllm_ascend.utils  # noqa: F401

        if not vllm_ascend.utils.enable_custom_op():
            raise RuntimeError("vllm_ascend custom ops disabled")
        op = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
        return (
            op(
                num_heads_q=_SAS_Q_HEADS,
                num_heads_kv=_SAS_KV_HEADS,
                head_dim=_SAS_HEAD_DIM,
                cu_seqlens_q=kw["cu_seqlens_q"],
                seqused_kv=kw["seqused_kv"],
                batch_size=1,
                max_seqlen_q=q_len,
                max_seqlen_kv=kv_len,
                cmp_topk=topk,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=127,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=cmp_ratio > 0,
                device=device,
            ),
            "vllm_ascend",
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        _log(f"  [warn] vllm_ascend metadata unavailable ({exc}); using zeros")
        return torch.zeros(1024, dtype=torch.int32, device=device), "zeros(fallback)"


def build_sparse_attn_sharedkv(case_id, device, metadata_mode="auto"):
    import torch

    mode, q_len, kv_len, cmp_ratio, topk = _SAS_SPECS[case_id]
    has_cmp = mode != "SWA"
    has_sparse = mode == "SCFA"

    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SAS_SEED)

    q = torch.rand(
        (q_len, _SAS_Q_HEADS, _SAS_HEAD_DIM),
        generator=gen,
        dtype=torch.float32,
        device="cpu",
    )
    q = (q * 20 - 10).to(torch.bfloat16).to(device)
    ori_kv, ori_block_table = _sas_paged(kv_len, gen, device)

    cmp_kv = cmp_block_table = cmp_sparse_indices = None
    if has_cmp:
        cmp_tokens = kv_len // cmp_ratio
        cmp_kv, cmp_block_table = _sas_paged(cmp_tokens, gen, device)
        if has_sparse:
            indices = torch.full(
                (q_len, _SAS_KV_HEADS, topk), -1, dtype=torch.int32, device="cpu"
            )
            for q_pos in range(q_len):
                visible = kv_len // cmp_ratio if q_len == 1 else (q_pos + 1) // cmp_ratio
                count = min(visible, topk)
                if count:
                    indices[q_pos, 0, :count] = torch.randperm(
                        visible, generator=gen, dtype=torch.int32, device="cpu"
                    )[:count]
            cmp_sparse_indices = indices.to(device)

    sinks = torch.rand(
        (_SAS_Q_HEADS,), generator=gen, dtype=torch.float32, device="cpu"
    ).to(device)

    kw = dict(
        q=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32, device=device),
        seqused_kv=torch.tensor([kv_len], dtype=torch.int32, device=device),
        sinks=sinks,
        softmax_scale=_SAS_SOFTMAX_SCALE,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )
    metadata, metadata_src = _sas_metadata(
        kw, q_len, kv_len, cmp_ratio, topk, device, metadata_mode
    )
    meta = dict(
        q_len=q_len,
        kv_len=kv_len,
        metadata=metadata,
        metadata_src=metadata_src,
    )
    return kw, meta


BUILDERS = {
    "causal_conv1d_fn": build_causal_conv1d_fn,
    "compressor": build_compressor,
    "gqa": build_gqa,
    "kv_rmsnorm_rope_cache": build_kv_rmsnorm_rope_cache,
    "lightning_indexer": build_lightning_indexer,
    "sparse_attn_sharedkv": build_sparse_attn_sharedkv,
}


# =========================================================================== #
# Calling the two implementations
# =========================================================================== #


def _require(module_path, attr):
    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr, None)
    if fn is None:
        raise AttributeError(f"{module_path}.{attr} not found")
    return fn


def _call_ref_causal_conv1d_fn(kw, meta):
    fn = _require(
        "flaggems_vllm.runtime.backend._ascend.ops.causal_conv1d_fn",
        "causal_conv1d_fn",
    )
    return fn(**kw)


def _call_ref_compressor(kw, meta):
    fn = _require(
        "flaggems_vllm.runtime.backend._ascend.ops.compressor", "compressor"
    )
    return fn(**kw)


def _patch_flash_api_cuda_literal():
    """Work around an upstream bug in the counterpart.

    ``flaggems_vllm/ops/flash_api.py::mha_fwd.dispatch()`` calls
    ``torch_device_fn.get_device_properties("cuda")`` -- a leftover from the
    CUDA original (``flag_gems/ops/flash_api.py`` has the same literal).  On
    Ascend that raises ``ValueError: Expected a npu device, but got: cuda`` and
    makes the whole counterpart unusable.  We retry without the bogus device
    and guarantee a ``multi_processor_count`` attribute (torch_npu exposes the
    core count under other names).

    Only used for the ``ref`` impl; the patch is process-local.
    """
    try:
        import flaggems_vllm.runtime as runtime
    except Exception:
        return None
    fn = getattr(runtime, "torch_device_fn", None)
    orig = getattr(fn, "get_device_properties", None)
    if orig is None or getattr(orig, "_commonir_bench_patched", False):
        return None

    class _Props:
        """Proxy that forwards everything and always offers multi_processor_count."""

        def __init__(self, inner):
            self._inner = inner
            self.multi_processor_count = int(
                getattr(inner, "multi_processor_count", 0)
                or getattr(inner, "vector_core_num", 0)
                or getattr(inner, "ai_core_num", 0)
                or 24
            )

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def patched(device=None, *a, **k):
        try:
            props = orig(device, *a, **k)
        except Exception:
            # the counterpart passed a literal "cuda"; ask for the default
            props = orig(*a, **k)
        return _Props(props)

    patched._commonir_bench_patched = True
    fn.get_device_properties = patched
    return "flash_api.dispatch(): get_device_properties('cuda') -> default device"


def _patch_flash_num_stages(limit=2):
    """Ascend Triton rejects ``tl.range(..., num_stages=N)`` with ``N > 2``.

    The counterpart's flash-attention kernels take ``num_stages`` from their
    tuned autotune configs (and, for the splitkv kernel, from a 3-valued
    heuristic), so on this toolchain they fail with
    ``AssertionError: Only `range` iterator supports num_stages <= 2``.
    Cap it.  Recorded in ``meta.notes``: the counterpart's numbers are then
    measured with the shallower pipeline.
    """
    touched = []

    def _visit(obj, label, depth=0):
        if obj is None or depth > 2:
            return
        cfgs = getattr(obj, "configs", None)
        if isinstance(cfgs, list) and cfgs and hasattr(cfgs[0], "num_stages"):
            new = []
            changed = False
            seen = set()
            for c in cfgs:
                ns = getattr(c, "num_stages", None)
                if isinstance(ns, int) and ns > limit:
                    try:
                        c = c._replace(num_stages=limit)
                        changed = True
                    except Exception:
                        pass
                # capping can collapse several entries (the tuned list
                # enumerates stages 2/3/4); drop exact duplicates
                key = repr(c)
                if key in seen:
                    continue
                seen.add(key)
                new.append(c)
            if changed:
                try:
                    obj.configs = new
                    touched.append(
                        f"{label}: {len(new)} autotune cfg (num_stages<={limit})"
                    )
                except Exception:
                    pass
        heur = None
        # triton.runtime.autotuner.Heuristics stores the dict as `.values`;
        # some forks call it `.heuristics`.  Handle both.
        for attr_name in ("values", "heuristics"):
            cand = getattr(obj, attr_name, None)
            if isinstance(cand, dict) and "num_stages" in cand:
                heur = cand
                break
        if heur is not None:
            heur["num_stages"] = (lambda args, _l=limit: _l)
            touched.append(f"{label}.num_stages -> {limit}")
        for inner in ("fn", "jit_function", "func"):
            _visit(getattr(obj, inner, None), f"{label}.{inner}", depth + 1)

    for modname in ("flaggems_vllm.ops.flash_kernel", "flag_gems.ops.flash_kernel"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr, obj in list(vars(mod).items()):
            if attr.startswith("__"):
                continue
            try:
                _visit(obj, f"{modname}.{attr}")
            except Exception:
                continue
    if not touched:
        return [
            "flash num_stages patch: no autotune configs/heuristics found, "
            "patch NOT applied"
        ]
    return touched


def _clamp_num_stages_globally(limit=2):
    """Clamp ``num_stages`` at the point every Triton kernel consumes it.

    The tuned flash-attention configs enumerate stages 2/3/4 and the splitkv
    kernel takes 3 from a heuristic; Ascend's ``tl.range`` rejects >2.  Patching
    the configs/heuristics objects by name proved fragile (the installed package
    need not match the repo layout), so clamp the kwarg in ``JITFunction.run``
    as well -- that covers every kernel regardless of how it was decorated.
    """
    try:
        import triton  # noqa: F401
    except Exception:
        return []
    candidates = []
    for modname in (
        "triton.runtime",
        "triton.runtime.jit",
        "triton.spec.ascend.runtime.jit",
        "triton.backends.ascend.runtime.jit",
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        cls = getattr(mod, "JITFunction", None)
        if cls is not None and cls not in candidates:
            candidates.append(cls)
    notes = []
    for cls in candidates:
        if getattr(cls.run, "_commonir_bench_clamped", False):
            continue
        orig = cls.run

        def run(self, *args, _orig=orig, **kwargs):
            v = kwargs.get("num_stages")
            if isinstance(v, int) and v > limit:
                kwargs["num_stages"] = limit
            return _orig(self, *args, **kwargs)

        run._commonir_bench_clamped = True
        cls.run = run
        notes.append(
            f"{cls.__module__}.{cls.__name__}.run: num_stages clamped to {limit}"
        )
    return notes


def _gqa_torch_golden(kw):
    """Torch composite golden for gqa.run(), used when the counterpart will not
    compile on this toolchain.

    ``is_causal=True`` in ``F.scaled_dot_product_attention`` is **upper-left**
    aligned, which is wrong for ``S_q != S_kv``; gqa.py (and flash-attn) are
    bottom-right aligned, i.e. ``CausalBias.causal_lower_right``.
    """
    import torch
    from torch.nn.attention.bias import causal_lower_right

    q = kw["query"].transpose(1, 2)
    k = kw["key"].transpose(1, 2)
    v = kw["value"].transpose(1, 2)
    seq_q, seq_kv, head_dim = q.shape[2], k.shape[2], q.shape[3]
    scale = kw["scaleValue"] if kw["scaleValue"] > 0 else float(head_dim) ** -0.5
    mask = causal_lower_right(seq_q, seq_kv) if kw["is_causal"] else None
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=True
    )
    return out.transpose(1, 2)


def _call_ref_gqa(kw, meta):
    """Drop-in for gqa.run() built on the FlashAttention forward stack.

    ``mha_fwd`` unpacks ``q.size()`` as ``(batch, seqlen_q, nheads, headdim)``,
    i.e. it takes BSND directly -- which is exactly ``gqa.run()``'s layout, so
    no transposes are needed.

    Semantics line up exactly: with ``is_causal=True`` the kernel uses
    ``col_rb = row + seqlen_k - seqlen_q`` (bottom-right aligned, identical to
    gqa's ``j <= i + (S_kv - S)``) and ``hid // h_hk_ratio`` (identical to
    gqa's ``hq // (N_q // N_kv)``); ``scale=None`` means ``1/sqrt(D)``.
    """
    fn = _require("flaggems_vllm.ops.attention", "flash_attention_forward")
    _note(_patch_flash_api_cuda_literal() or "flash_api cuda literal: nothing to patch")
    for msg in _patch_flash_num_stages(2) + _clamp_num_stages_globally(2):
        _note(msg)
    # The split-kv dispatcher needs `get_device_properties(...).multi_processor_count`
    # (torch_npu does not expose it) and its kernel's num_stages heuristic is 3.
    # Force the plain forward kernel instead.
    _note("gqa ref launched with disable_splitkv=True (splitkv is CUDA-only)")
    q = kw["query"]
    k = kw["key"]
    v = kw["value"]
    seq_q, seq_kv, head_dim = q.shape[1], k.shape[1], q.shape[3]
    scale = kw["scaleValue"] if kw["scaleValue"] > 0 else float(head_dim) ** -0.5
    try:
        return fn(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            None,
            None,
            seq_q,
            seq_kv,
            0.0,
            bool(kw["is_causal"]),
            False,
            scale=scale,
            disable_splitkv=True,
        )[0]
    except Exception as exc:
        if "ompil" not in type(exc).__name__ and "ompil" not in str(exc)[:400]:
            raise
        _note(
            "gqa ref FALLBACK: flash_attention_forward does not compile on this "
            f"toolchain ({type(exc).__name__}); compared against "
            "F.scaled_dot_product_attention(causal_lower_right, enable_gqa=True) "
            "-- a vendor SDPA golden, so the ref timing is NOT a Triton kernel"
        )
        meta["ref_is_torch_golden"] = True
        return _gqa_torch_golden(kw)


def _call_ref_kv_rmsnorm_rope_cache(kw, meta):
    fn = _require(
        "flaggems_vllm.runtime.backend._ascend.ops.kv_rmsnorm_rope_cache",
        "kv_rmsnorm_rope_cache",
    )
    return fn(**kw)


def _call_ref_lightning_indexer(kw, meta):
    fn = _require(
        "flaggems_vllm.runtime.backend._ascend.ops.lightning_indexer",
        "lightning_indexer",
    )
    return fn(**kw)


def _call_ref_sparse_attn_sharedkv(kw, meta):
    fn = _require(
        "flaggems_vllm.runtime.backend._ascend.ops.sparse_attn_sharedkv",
        "sparse_attn_sharedkv",
    )
    extra = dict(kw)
    q = extra.pop("q")
    extra["metadata"] = meta["metadata"]
    return fn(q, host_q_lens=(meta["q_len"],), host_kv_lens=(meta["kv_len"],), **extra)


REF_CALLERS = {
    "causal_conv1d_fn": _call_ref_causal_conv1d_fn,
    "compressor": _call_ref_compressor,
    "gqa": _call_ref_gqa,
    "kv_rmsnorm_rope_cache": _call_ref_kv_rmsnorm_rope_cache,
    "lightning_indexer": _call_ref_lightning_indexer,
    "sparse_attn_sharedkv": _call_ref_sparse_attn_sharedkv,
}

REF_SOURCE = {
    "causal_conv1d_fn": "FlagGems-vllm/_ascend/ops/causal_conv1d_fn.py::causal_conv1d_fn",
    "compressor": "FlagGems-vllm/_ascend/ops/compressor.py::compressor",
    "gqa": "FlagGems-vllm/ops/attention.py::flash_attention_forward (BHSD, "
    "flaggems/ops/flash_api.py::mha_fwd + flash_kernel.py)",
    "kv_rmsnorm_rope_cache": "FlagGems-vllm/_ascend/ops/kv_rmsnorm_rope_cache.py::"
    "kv_rmsnorm_rope_cache",
    "lightning_indexer": "FlagGems-vllm/_ascend/ops/lightning_indexer.py::lightning_indexer",
    "sparse_attn_sharedkv": "FlagGems-vllm/_ascend/ops/sparse_attn_sharedkv.py::"
    "sparse_attn_sharedkv",
}


def call_new(op, kw, meta, kernel_dir, consts=None):
    mod = _load_kernel_module(op, kernel_dir)
    for name, value in (consts or {}).items():
        if not hasattr(mod, name):
            raise AttributeError(
                f"{op}.py has no module-level constant {name!r} "
                f"(available: {[a for a in dir(mod) if a.isupper()]})"
            )
        setattr(mod, name, value)
    return mod.run(**kw)


def _parse_consts(spec):
    """``_BLOCK_N=64,_BLOCK_M=32`` -> {'_BLOCK_N': 64, '_BLOCK_M': 32}."""
    out = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--new-const expects NAME=VALUE, got {item!r}")
        name, raw = item.split("=", 1)
        raw = raw.strip()
        for conv in (int, float):
            try:
                out[name.strip()] = conv(raw)
                break
            except ValueError:
                continue
        else:
            out[name.strip()] = raw
    return out


# =========================================================================== #
# Artifact collection
# =========================================================================== #
#
# Artifacts are plain CPU tensors (plus a little metadata) so they can be
# compared on a machine without an NPU.  Tensors mutated in place by the
# operator (paged caches, conv states) are gathered down to the touched rows.


def _torch_load(path):
    """``weights_only`` default flipped in torch 2.6; older torch lacks it."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def collect_causal_conv1d_fn(kw, meta, result):
    import torch

    slots = torch.tensor(meta["slots"], dtype=torch.long, device=kw["conv_states"].device)
    return dict(
        out=result.detach(),
        conv_states=kw["conv_states"].index_select(0, slots).detach(),
        lives=meta["live"],
        seqlens=meta["seqlens"],
    )


def collect_compressor(kw, meta, result):
    return dict(
        out=result.detach(),
        # detach() keeps sharing storage; .to("cpu") in the worker is the copy
        state_cache=kw["state_cache"].detach(),
        valid_rows=meta["valid_rows"],
        rank3=meta["rank3"],
    )


def collect_gqa(kw, meta, result):
    return dict(out=result.detach())


def collect_kv_rmsnorm_rope_cache(kw, meta, result):
    import torch

    kc = kw["k_cache"]
    cc = kw["ckv_cache"]
    rope = meta["rope"]
    rms = meta["rms"]
    slot = kw["index"].to(torch.long).reshape(-1)
    return dict(
        k_rope=result[2].detach(),
        c_kv=result[3].detach(),
        k_cache=kc.view(-1, rope).index_select(0, slot).detach(),
        ckv_cache=cc.view(-1, rms).index_select(0, slot).detach(),
    )


def collect_lightning_indexer(kw, meta, result):
    return dict(indices=result[0].detach())


def collect_sparse_attn_sharedkv(kw, meta, result):
    return dict(out=result[0].detach(), lse=result[1].detach())


COLLECTORS = {
    "causal_conv1d_fn": collect_causal_conv1d_fn,
    "compressor": collect_compressor,
    "gqa": collect_gqa,
    "kv_rmsnorm_rope_cache": collect_kv_rmsnorm_rope_cache,
    "lightning_indexer": collect_lightning_indexer,
    "sparse_attn_sharedkv": collect_sparse_attn_sharedkv,
}


# =========================================================================== #
# Comparison
# =========================================================================== #

COMPARE_MODE = {
    "causal_conv1d_fn": "tensor",  # special-cased for the pad-slot compaction
    "compressor": "tensor",
    "gqa": "tensor",
    "kv_rmsnorm_rope_cache": "tensor",
    "lightning_indexer": "index_multiset",
    "sparse_attn_sharedkv": "tensor",
}

TOLERANCE = {
    # (rtol, atol) taken from the counterpart repo's own test suites
    "causal_conv1d_fn": (1e-2, 5e-2),
    "compressor": (5e-2, 5e-2),
    "gqa": (2e-2, 2e-2),
    "kv_rmsnorm_rope_cache": (1e-3, 1e-2),
    "sparse_attn_sharedkv": (2e-2, 2e-2),
}


def _tensor_stats(a, b, chunk=1 << 22):
    """Diff statistics for two same-shape CPU tensors.

    Streamed in chunks so that a 256M-element bf16 tensor never materialises a
    full float32 copy (that is ~1 GB per side).
    """
    import torch

    fa = a.reshape(-1)
    fb = b.reshape(-1)
    n = int(fa.numel())
    if n == 0:
        return dict(
            shape=list(a.shape),
            dtype=str(a.dtype),
            numel=0,
            max_abs_err=0.0,
            mean_abs_err=0.0,
            max_rel_err=0.0,
            rms_ref=0.0,
            cosine=float("nan"),
            nan_new=0,
            nan_ref=0,
        )

    mx = 0.0
    sabs = 0.0
    mre = 0.0
    sa = sb = saa = sbb = sab = 0.0
    nan_new = nan_ref = 0
    for i in range(0, n, chunk):
        xa = fa[i : i + chunk].float()
        xb = fb[i : i + chunk].float()
        nan_new += int(torch.isnan(xa).sum())
        nan_ref += int(torch.isnan(xb).sum())
        d = (xa - xb).abs()
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        mx = max(mx, float(d.max()))
        sabs += float(d.sum(dtype=torch.float64))
        mre = max(mre, float((d / xb.abs().clamp_min(1e-12)).max()))
        # accumulate in fp64 without materialising a float64 copy of the chunk
        sa += float(xa.sum(dtype=torch.float64))
        sb += float(xb.sum(dtype=torch.float64))
        saa += float((xa * xa).sum(dtype=torch.float64))
        sbb += float((xb * xb).sum(dtype=torch.float64))
        sab += float((xa * xb).sum(dtype=torch.float64))

    # percentiles of |err| via a second (histogram) pass -- exact sorting of a
    # 256M-element vector is not worth it for a report field
    pct = dict(p50=None, p90=None, p99=None)
    if mx > 0:
        bins = 512
        hist = torch.zeros(bins, dtype=torch.float64)
        for i in range(0, n, chunk):
            d = (fa[i : i + chunk].float() - fb[i : i + chunk].float()).abs()
            d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            idx = (d.double() * (bins / mx)).to(torch.long).clamp_(0, bins - 1)
            hist += torch.bincount(idx, minlength=bins).double()
        cum = torch.cumsum(hist, 0) / n
        # value at the bin centre of each percentile
        for key, q in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
            b_idx = int(torch.searchsorted(cum, torch.tensor(q, dtype=torch.float64)))
            pct[key] = round((min(b_idx, bins - 1) + 0.5) * mx / bins, 6)

    var_a = n * saa - sa * sa
    var_b = n * sbb - sb * sb
    cov = n * sab - sa * sb
    cos = (cov / math.sqrt(var_a * var_b)) if var_a > 0 and var_b > 0 else float("nan")

    return dict(
        shape=list(a.shape),
        dtype=str(a.dtype),
        numel=n,
        max_abs_err=mx,
        mean_abs_err=sabs / n,
        max_rel_err=mre,
        p50_abs_err=pct["p50"],
        p90_abs_err=pct["p90"],
        p99_abs_err=pct["p99"],
        rms_ref=math.sqrt(sbb / n),
        cosine=cos,
        nan_new=nan_new,
        nan_ref=nan_ref,
    )


def _worst_elements(a, b, n_worst, budget=1 << 23):
    """Locations of the n_worst largest |err| elements (approximate if huge).

    This is what replaces the raw tensor for debugging: a few hundred bytes of
    coordinates + values per tensor instead of hundreds of megabytes.
    """
    import torch

    fa = a.reshape(-1)
    fb = b.reshape(-1)
    total = int(fa.numel())
    if total == 0:
        return [], False
    step = max(1, total // budget)
    out, approx = [], step > 1
    if approx:
        # coarse pass: worst coarse block, then refine inside it
        block = 8112 * step
        blocks = (total + block - 1) // block
        best, best_err = 0, -1.0
        for blk in range(blocks):
            lo, hi = blk * block, min((blk + 1) * block, total)
            d = (
                (fa[lo:hi].float() - fb[lo:hi].float())
                .abs()
                .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            )
            m = float(d.max())
            if m > best_err:
                best_err, best = m, lo
        lo = best
        hi = min(best + block, total)
    else:
        lo, hi = 0, total

    d = (
        (fa[lo:hi].float() - fb[lo:hi].float())
        .abs()
        .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    )
    k = min(n_worst, d.numel())
    if k <= 0:
        return [], approx
    vals, idx = torch.topk(d, k)
    for v, i in zip(vals.tolist(), idx.tolist()):
        flat = lo + i
        out.append(
            dict(
                flat_index=int(flat),
                abs_err=round(float(v), 8),
                new=float(fa[flat]),
                ref=float(fb[flat]),
            )
        )
    return out, approx


def _cmp_tensor_block(name, a, b, rtol, atol, digest_n=16):
    import torch

    a = a.detach().cpu()
    b = b.detach().cpu()
    if tuple(a.shape) != tuple(b.shape):
        return dict(
            tensor=name,
            ok=False,
            error=f"shape mismatch new={tuple(a.shape)} ref={tuple(b.shape)}",
        )
    st = _tensor_stats(a, b)
    ok = bool(
        torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol, equal_nan=True)
    )
    st.update(tensor=name, ok=ok, rtol=rtol, atol=atol)
    if not ok and digest_n:
        worst, approx = _worst_elements(a, b, digest_n)
        st["worst"] = worst
        st["worst_is_approximate"] = approx
    return st


def _cmp_index_multiset(a, b):
    """Compare top-k index rows as multisets (order is free), ignoring -1 pad."""
    import torch

    a = a.detach().cpu().to(torch.int64)
    b = b.detach().cpu().to(torch.int64)
    if tuple(a.shape) != tuple(b.shape):
        return dict(
            tensor="indices",
            ok=False,
            error=f"shape mismatch new={tuple(a.shape)} ref={tuple(b.shape)}",
        )
    flat_a = a.reshape(a.shape[0], -1)
    flat_b = b.reshape(b.shape[0], -1)
    rows = flat_a.shape[0]
    bad, pad_mismatch, bad_example, trivial = 0, 0, None, 0
    for r in range(rows):
        ra = sorted(int(v) for v in flat_a[r].tolist() if v != -1)
        rb = sorted(int(v) for v in flat_b[r].tolist() if v != -1)
        # a row that is exactly 0,1,2,... is the degenerate "visible window
        # <= sparse_count" branch, where neither impl does any selection
        if rb == list(range(len(rb))):
            trivial += 1
        if ra != rb:
            bad += 1
            if bad_example is None:
                only_a = sorted(set(ra) - set(rb))[:8]
                only_b = sorted(set(rb) - set(ra))[:8]
                bad_example = dict(row=r, only_new=only_a, only_ref=only_b)
        if int((flat_a[r] == -1).sum()) != int((flat_b[r] == -1).sum()):
            pad_mismatch += 1
    return dict(
        tensor="indices",
        ok=bad == 0 and pad_mismatch == 0,
        numel=int(a.numel()),
        rows=int(rows),
        mismatched_rows=bad,
        padcount_mismatched_rows=pad_mismatch,
        first_mismatch=bad_example,
        trivial_rows=trivial,
        all_rows_trivial=trivial == rows,
    )


def compare_case(op, outdir, case_id, digest_n=16, new_tag="", ref_tag=""):
    import torch

    a_path = _art_path(outdir, op, case_id, NEW_IMPL, new_tag)
    b_path = _art_path(outdir, op, case_id, REF_IMPL, ref_tag)
    if not (os.path.isfile(a_path) and os.path.isfile(b_path)):
        return dict(op=op, case=case_id, status="skipped", reason="missing artifact")
    try:
        new = _torch_load(a_path)
        ref = _torch_load(b_path)
    except Exception as exc:
        return dict(op=op, case=case_id, status="error", reason=f"load: {exc}")

    mode = COMPARE_MODE[op]
    rtol, atol = TOLERANCE.get(op, (1e-2, 1e-2))
    checks = []

    if mode == "index_multiset":
        checks.append(_cmp_index_multiset(new["indices"], ref["indices"]))

    elif op == "causal_conv1d_fn":
        # The CommonIR op concatenates only the live sequences (pad_slot_id
        # sequences contribute no output column), while the counterpart keeps
        # one column per input token.  Pick whichever reference layout matches
        # the reference tensor we actually got, then compare live columns and
        # the touched rows of conv_states.
        out_new = new["out"]
        out_ref = ref["out"]
        lives = new["lives"]
        seqlens = new["seqlens"]
        live_cols = sum(n for n, live in zip(seqlens, lives) if live)
        if int(out_ref.shape[1]) == live_cols:
            ref_cat = out_ref  # counterpart already compacted
        else:
            pos, cols = 0, []
            for i, live in enumerate(lives):
                n = seqlens[i]
                if live:
                    cols.append(out_ref[:, pos : pos + n])
                pos += n
            ref_cat = torch.cat(cols, dim=1) if cols else out_ref[:, :0]
        checks.append(_cmp_tensor_block("out", out_new, ref_cat, rtol, atol, digest_n))
        checks.append(
            _cmp_tensor_block("conv_states", new["conv_states"], ref["conv_states"], rtol, atol, digest_n)
        )

    elif op == "compressor":
        rows = new["valid_rows"]
        a = new["out"].reshape(-1, new["out"].shape[-1])
        b = ref["out"].reshape(-1, ref["out"].shape[-1])
        if new["rank3"]:
            checks.append(
                _cmp_tensor_block("out", a[: sum(rows)], b[: sum(rows)], rtol, atol, digest_n)
            )
        else:
            total = sum(rows)
            checks.append(_cmp_tensor_block("out", a[:total], b[:total], rtol, atol, digest_n))
        checks.append(
            _cmp_tensor_block(
                "state_cache",
                new["state_cache"],
                ref["state_cache"],
                rtol,
                atol,
                digest_n,
            )
        )

    elif op == "kv_rmsnorm_rope_cache":
        for key in ("k_rope", "c_kv", "k_cache", "ckv_cache"):
            checks.append(_cmp_tensor_block(key, new[key], ref[key], rtol, atol, digest_n))

    elif op == "sparse_attn_sharedkv":
        checks.append(_cmp_tensor_block("out", new["out"], ref["out"], rtol, atol, digest_n))
        checks.append(
            dict(
                tensor="lse",
                ok=tuple(new["lse"].shape) == tuple(ref["lse"].shape),
                new_numel=int(new["lse"].numel()),
                ref_numel=int(ref["lse"].numel()),
            )
        )

    else:  # gqa and any plain tensor op
        checks.append(_cmp_tensor_block("out", new["out"], ref["out"], rtol, atol, digest_n))

    ok = all(bool(c.get("ok")) for c in checks)
    return dict(op=op, case=case_id, status="ok" if ok else "MISMATCH", checks=checks)


# =========================================================================== #
# Worker
# =========================================================================== #


def _measure_loop(fn, device, warmup, iters, rounds):
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(rounds):
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) / iters * 1e3)
    samples.sort()
    n = len(samples)
    return dict(
        mode="loop",
        iters=iters,
        rounds=rounds,
        median_ms=samples[n // 2],
        min_ms=samples[0],
        mean_ms=sum(samples) / n,
        samples_ms=samples,
    )


def _jit_classes():
    out = []
    for modname in ("triton.runtime", "triton.runtime.jit",
                    "triton.spec.ascend.runtime.jit",
                    "triton.backends.ascend.runtime.jit"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        cls = getattr(mod, "JITFunction", None)
        if cls is not None and cls not in out:
            out.append(cls)
    return out


def _measure_host_split(fn, device, warmup=3, iters=20, rounds=5):
    """Split a call into host bookkeeping vs device work.

    `npu-graph` cannot capture an op that synchronises the host (both
    implementations of causal_conv1d_fn do `.cpu().tolist()` in run()), so for
    those the only honest attribution is: time run() normally, then time it
    again with every Triton launch stubbed out.  The second number is the host
    cost *including* the D2H sync; device = total - host.
    """
    total = _measure_loop(fn, device, warmup, iters, rounds)

    classes = _jit_classes()
    saved = [(cls, cls.run) for cls in classes]

    def stub_run(self, *a, **k):
        return None

    for cls, _orig in saved:
        cls.run = stub_run
    try:
        host = _measure_loop(fn, device, warmup, iters, rounds)
    finally:
        for cls, orig in saved:
            cls.run = orig

    return dict(
        mode="host-split",
        iters=iters,
        rounds=rounds,
        median_ms=total["median_ms"],
        min_ms=total["min_ms"],
        mean_ms=total["mean_ms"],
        host_ms=host["median_ms"],
        device_ms=max(total["median_ms"] - host["median_ms"], 0.0),
        total_samples_ms=total["samples_ms"],
        host_samples_ms=host["samples_ms"],
    )


def _measure_do_bench(fn, device):
    ms = None
    if str(device).startswith("npu"):
        try:
            import triton.backends.ascend.testing as ascend_testing

            fn_bench = getattr(ascend_testing, "do_bench_npu", None)
            if fn_bench is not None:
                ms = fn_bench(fn)
        except Exception:
            ms = None
    if ms is None:
        import triton

        ms = triton.testing.do_bench(fn, return_mode="median")
    return dict(mode="do_bench", median_ms=float(ms))


def _measure_graph(fn, device, warmup=3, iters=20, rounds=5):
    """Replay-only timing: strips *all* host overhead from the number.

    The counterpart rebuilds its launch metadata on every call (~1.0 ms for
    causal_conv1d_fn, ~1.5 ms for compressor), which completely hides its kernel
    time.  Capturing the call once and timing `graph.replay()` leaves pure
    device time, so the two implementations can be compared like for like.
    """
    import torch

    mod = torch.npu if str(device).startswith("npu") else torch.cuda
    graph_cls = getattr(mod, "NPUGraph", None) or getattr(mod, "CUDAGraph", None)
    if graph_cls is None or not hasattr(mod, "graph"):
        raise RuntimeError(f"{device} backend exposes no capture API")
    side = mod.Stream()
    side.wait_stream(mod.current_stream())
    with mod.stream(side):
        for _ in range(warmup):
            fn()
    mod.current_stream().wait_stream(side)
    mod.synchronize()
    graph = graph_cls()
    with mod.graph(graph):
        fn()
    mod.synchronize()
    samples = []
    for _ in range(rounds):
        mod.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        mod.synchronize()
        samples.append((time.perf_counter() - t0) / iters * 1e3)
    samples.sort()
    n = len(samples)
    return dict(
        mode="graph",
        iters=iters,
        rounds=rounds,
        median_ms=samples[n // 2],
        min_ms=samples[0],
        mean_ms=sum(samples) / n,
        samples_ms=samples,
    )


def worker_main(args):
    import torch

    op = args.op
    case_id = args.case
    impl = args.impl
    device = args.device
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
    tag = getattr(args, "impl_tag", "") or ""
    if impl == REF_IMPL and op == "compressor":
        # let the counterpart print its resource-modelled BLOCK_K into logs/
        os.environ["COMPRESSOR_PRINT_RESOURCE_PLAN"] = "1"

    if device.startswith("npu"):
        # The CommonIR kernels rely on torch_npu having registered the "npu"
        # device; the flagtree success_case modules never import it themselves.
        try:
            import torch_npu  # noqa: F401

            torch.npu.set_device(_device_index(device))
        except Exception as exc:  # pragma: no cover - env dependent
            out = dict(op=op, case=case_id, impl=impl, device=device, status="error")
            out["error"] = f"torch_npu unavailable for device '{device}': {exc}"
            _json_dump(_raw_path(args.outdir, op, case_id, impl, tag), out)
            return 1

    out = dict(op=op, case=case_id, impl=impl, device=device, status="ok")
    try:
        if op == "sparse_attn_sharedkv":
            kw, meta = BUILDERS[op](case_id, device, metadata_mode=args.sparse_metadata)
        else:
            kw, meta = BUILDERS[op](case_id, device)

        consts = _parse_consts(getattr(args, "new_const", None))
        if impl == NEW_IMPL:
            def invoke():
                return call_new(op, kw, meta, args.kernel_dir, consts)
        else:
            def invoke():
                return REF_CALLERS[op](kw, meta)

        # ---- accuracy pass: one clean call, artifacts moved to CPU --------
        with torch.inference_mode():
            _sync(device)
            result = invoke()
            _sync(device)
            artifacts = COLLECTORS[op](kw, meta, result)
            cpu_artifacts = {
                k: (v.detach().to("cpu") if torch.is_tensor(v) else v)
                for k, v in artifacts.items()
            }
        if args.save_artifacts:
            path = _art_path(args.outdir, op, case_id, impl, tag)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(cpu_artifacts, path)
            out["artifact"] = path
            out["artifact_mb"] = round(os.path.getsize(path) / 1e6, 2)
        del artifacts, cpu_artifacts, result
        if device.startswith("npu") and hasattr(torch, "npu"):
            torch.npu.empty_cache()

        # ---- performance pass -------------------------------------------
        if args.no_bench:
            out["bench"] = dict(mode="skipped")
        else:
            with torch.inference_mode():
                if args.measure_mode == "host-split":
                    bench = _measure_host_split(
                        invoke, device, warmup=args.warmup,
                        iters=args.iters, rounds=args.rounds,
                    )
                elif args.measure_mode == "npu-graph":
                    try:
                        bench = _measure_graph(
                            invoke, device, warmup=args.warmup,
                            iters=args.iters, rounds=args.rounds,
                        )
                    except Exception as exc:
                        _note(f"graph capture failed ({type(exc).__name__}: {exc}); "
                              f"fell back to loop timing")
                        bench = _measure_loop(
                            invoke, device, warmup=args.warmup,
                            iters=args.iters, rounds=args.rounds,
                        )
                        bench["graph_capture_failed"] = str(exc)[:200]
                elif args.measure_mode == "do_bench":
                    bench = _measure_do_bench(invoke, device)
                else:
                    bench = _measure_loop(
                        invoke,
                        device,
                        warmup=args.warmup,
                        iters=args.iters,
                        rounds=args.rounds,
                    )
            if device.startswith("npu") and hasattr(torch, "npu"):
                try:
                    bench["peak_alloc_mb"] = round(
                        torch.npu.max_memory_allocated() / 2**20, 2
                    )
                except Exception:
                    pass
            out["bench"] = bench

        out["meta"] = {
            k: v for k, v in meta.items() if k != "metadata"
        }
        out["env"] = {
            "torch": torch.__version__,
            "host": platform.node(),
        }
        try:
            import triton

            out["env"]["triton"] = triton.__version__
        except Exception:
            pass
        try:
            import torch_npu  # noqa: F401

            out["env"]["torch_npu"] = getattr(torch_npu, "__version__", "?")
        except Exception:
            pass

    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        _log(out["traceback"])

    # workarounds must be visible even when the call failed -- otherwise a
    # silently-unapplied patch looks exactly like a broken counterpart
    if _RUN_NOTES:
        out["notes"] = list(_RUN_NOTES)
    _json_dump(_raw_path(args.outdir, op, case_id, impl, tag), out)
    return 0 if out["status"] == "ok" else 1


# =========================================================================== #
# Driver
# =========================================================================== #


def _resolve_cases(ops, case_set, explicit):
    resolved = {}
    for op in ops:
        if explicit:
            wanted = []
            for token in explicit.split(","):
                token = token.strip()
                if not token:
                    continue
                if token in CASES[op]:
                    wanted.extend(CASES[op][token])
                else:
                    wanted.append(token)
            seen, ordered = set(), []
            for c in wanted:
                if c not in seen:
                    seen.add(c)
                    ordered.append(c)
            unknown = [c for c in ordered if c not in _case_ids(op)]
            if unknown:
                raise SystemExit(
                    f"unknown case(s) for {op}: {unknown}\n"
                    f"available: {sorted(_case_ids(op))}"
                )
            resolved[op] = ordered
        else:
            resolved[op] = list(CASES[op][case_set])
    return resolved


def _case_ids(op):
    if op == "causal_conv1d_fn":
        return _CC1D_SPECS
    if op == "compressor":
        return _COMPRESSOR_SPECS
    if op == "gqa":
        return _GQA_SPECS
    if op == "kv_rmsnorm_rope_cache":
        return _KV_SPECS
    if op == "lightning_indexer":
        return _LI_SPECS
    if op == "sparse_attn_sharedkv":
        return _SAS_SPECS
    raise KeyError(op)


def _worker_env(args, op, impl):
    env = dict(os.environ)
    env["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.dump_ir:
        dump_dir = os.path.abspath(
            os.path.join(args.outdir, "ir", op, impl)
        )
        os.makedirs(dump_dir, exist_ok=True)
        env["TRITON_KERNEL_DUMP"] = "1"
        env["TRITON_ALWAYS_COMPILE"] = "1"
        env["TRITON_DUMP_DIR"] = dump_dir
    if args.dump_pass_ir:
        env["MLIR_ENABLE_DUMP"] = args.dump_pass_ir
    return env


def produce(outdir, ops, cases, args):
    _log("=" * 78)
    _log(f"PRODUCE  device={args.device}  outdir={outdir}")
    _log("=" * 78)
    results = []
    _case_rows = {}
    for op in ops:
        for case_id in cases[op]:
            for impl in IMPLS:
                tag = f"{op}__{case_id}__{impl}"
                log_path = _log_path(outdir, op, case_id, impl)
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                cmd = [
                    sys.executable,
                    _SELF,
                    "worker",
                    "--outdir",
                    outdir,
                    "--op",
                    op,
                    "--case",
                    case_id,
                    "--impl",
                    impl,
                    "--kernel-dir",
                    args.kernel_dir,
                    "--device",
                    args.device,
                    "--measure-mode",
                    args.measure_mode,
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--rounds",
                    str(args.rounds),
                    "--sparse-metadata",
                    args.sparse_metadata,
                ]
                if getattr(args, "new_const", None):
                    cmd += ["--new-const", args.new_const]
                if args.no_bench:
                    cmd.append("--no-bench")
                if not args.save_artifacts:
                    cmd.append("--no-save-artifacts")
                t0 = time.time()
                _log(f"  [{tag}] ... ")
                with open(log_path, "w") as lf:
                    try:
                        proc = subprocess.run(
                            cmd,
                            env=_worker_env(args, op, impl),
                            stdout=lf,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout,
                        )
                        rc = proc.returncode
                    except subprocess.TimeoutExpired:
                        rc = 124
                dt = time.time() - t0
                raw = _json_load(_raw_path(outdir, op, case_id, impl), {}) or {}
                status = raw.get("status", "crash" if rc not in (0,) else "unknown")
                if rc == 124:
                    status = "timeout"
                bench = raw.get("bench") or {}
                lat = bench.get("median_ms")
                lat_s = f"{lat:8.3f} ms" if isinstance(lat, (int, float)) else "     n/a "
                _log(f"       -> {status:<9} {lat_s}  ({dt:6.1f}s wall)  log={log_path}")
                if status == "error":
                    _log(f"       !! {raw.get('error')}")
                _case_rows.setdefault((op, case_id), {})[impl] = len(results)
                results.append(
                    dict(
                        op=op,
                        case=case_id,
                        impl=impl,
                        status=status,
                        rc=rc,
                        wall_s=round(dt, 2),
                        median_ms=lat,
                        min_ms=bench.get("min_ms"),
                        measure_mode=bench.get("mode"),
                        host_ms=bench.get("host_ms"),
                        device_ms=bench.get("device_ms"),
                        artifact_mb=raw.get("artifact_mb"),
                        peak_alloc_mb=bench.get("peak_alloc_mb"),
                        notes=raw.get("notes")
                        or raw.get("meta", {}).get("notes"),
                        log=os.path.relpath(log_path, outdir),
                    )
                )
    _reconcile_modes(args, outdir, results, _case_rows)
    _json_dump(os.path.join(outdir, "produce.json"), results)
    return results


def _respawn_single(args, op, case_id, impl):
    """Re-run one (op, case, impl) worker with the current args."""
    log_path = _log_path(args.outdir, op, case_id, impl)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [
        sys.executable, _SELF, "worker",
        "--outdir", args.outdir, "--op", op, "--case", case_id, "--impl", impl,
        "--kernel-dir", args.kernel_dir, "--device", args.device,
        "--measure-mode", args.measure_mode,
        "--warmup", str(args.warmup), "--iters", str(args.iters),
        "--rounds", str(args.rounds),
        "--sparse-metadata", args.sparse_metadata,
    ]
    if getattr(args, "new_const", None):
        cmd += ["--new-const", args.new_const]
    if args.no_bench:
        cmd.append("--no-bench")
    if not args.save_artifacts:
        cmd.append("--no-save-artifacts")
    try:
        with open(log_path, "w") as lf:
            subprocess.run(
                cmd, env=_worker_env(args, op, impl), stdout=lf,
                stderr=subprocess.STDOUT, timeout=args.timeout,
            )
    except subprocess.TimeoutExpired:
        pass


def _reconcile_modes(args, outdir, results, case_rows):
    """A ratio across 'device-only' and 'host+device' is meaningless.

    When one side of a case cannot be graph-captured (most counterparts sync
    the host inside the call), re-measure the capturable side in loop mode so
    both numbers are the same kind of number.
    """
    if getattr(args, "measure_mode", None) != "npu-graph":
        return
    for (op, case_id), rows in case_rows.items():
        modes = {impl: results[i].get("measure_mode") for impl, i in rows.items()}
        if len(set(modes.values())) < 2:
            continue
        for impl, idx in rows.items():
            if modes[impl] != "graph":
                continue
            _log(f"  {op}/{case_id}: {impl} captured, other side did not "
                 f"-> re-measuring {impl} in loop mode")
            keep = args.measure_mode
            args.measure_mode = "loop"
            _respawn_single(args, op, case_id, impl)
            args.measure_mode = keep
            raw = _json_load(_raw_path(outdir, op, case_id, impl), {}) or {}
            bench = raw.get("bench") or {}
            row = results[idx]
            for key, val in (
                ("median_ms", bench.get("median_ms")),
                ("min_ms", bench.get("min_ms")),
                ("measure_mode", bench.get("mode")),
                ("host_ms", bench.get("host_ms")),
                ("device_ms", bench.get("device_ms")),
            ):
                row[key] = val
            row["status"] = raw.get("status", row["status"])


def _expand_sweep(spec):
    """'a=1|2,b=3|4' -> [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, ...]."""
    knobs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--sweep expects NAME=V1|V2, got {item!r}")
        name, raw = item.split("=", 1)
        parsed = []
        for v in raw.split("|"):
            v = v.strip()
            for conv in (int, float):
                try:
                    parsed.append(conv(v))
                    break
                except ValueError:
                    continue
            else:
                parsed.append(v)
        knobs.append((name.strip(), parsed))
    variants = [{}]
    for name, values in knobs:
        variants = [dict(v, **{name: val}) for v in variants for val in values]
    return variants


def _spawn_worker(args, op, case_id, impl, tag, new_const=None):
    log_path = _log_path(args.outdir, op, case_id, impl, tag)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [
        sys.executable, _SELF, "worker",
        "--outdir", args.outdir, "--op", op, "--case", case_id,
        "--impl", impl, "--impl-tag", tag,
        "--kernel-dir", args.kernel_dir, "--device", args.device,
        "--measure-mode", args.measure_mode,
        "--warmup", str(args.warmup), "--iters", str(args.iters),
        "--rounds", str(args.rounds),
        "--sparse-metadata", args.sparse_metadata,
    ]
    if new_const:
        cmd += ["--new-const", new_const]
    if args.no_bench:
        cmd.append("--no-bench")
    if not args.save_artifacts:
        cmd.append("--no-save-artifacts")
    try:
        with open(log_path, "w") as lf:
            rc = subprocess.run(
                cmd, env=_worker_env(args, op, impl), stdout=lf,
                stderr=subprocess.STDOUT, timeout=args.timeout,
            ).returncode
    except subprocess.TimeoutExpired:
        rc = 124
    raw = _json_load(_raw_path(args.outdir, op, case_id, impl, tag), {}) or {}
    if rc == 124:
        raw["status"] = "timeout"
    raw.setdefault("status", "crash" if rc else "unknown")
    return raw


def sweep(args, ops, cases):
    variants = _expand_sweep(args.sweep)
    _log("=" * 78)
    _log(f"SWEEP  {len(variants)} variant(s) x cases   measure={args.measure_mode}")
    for i, v in enumerate(variants):
        _log(f"  v{i}: {v}")
    _log("=" * 78)
    rows = []
    for op in ops:
        for case_id in cases[op]:
            ref = _spawn_worker(args, op, case_id, REF_IMPL, "base")
            ref_bench = ref.get("bench") or {}
            ref_ms = ref_bench.get("median_ms")
            ref_mode = ref_bench.get("mode")
            _log(f"  {op}/{case_id}: ref {ref_ms if ref_ms else ref['status']}"
                 f" (mode={ref_mode})")
            # a ratio is only meaningful if both sides were measured the same
            # way; if the counterpart cannot be graph-captured, drop everyone
            # back to loop timing for this case
            mode = getattr(args, "measure_mode", "loop")
            if mode == "npu-graph" and ref_mode != "graph":
                _log("    (ref not capturable -> measuring this case in loop mode)")
                mode = "loop"
            args.measure_mode = mode
            for i, variant in enumerate(variants):
                tag = f"v{i}"
                const = ",".join(f"{k}={v}" for k, v in variant.items())
                new = _spawn_worker(args, op, case_id, NEW_IMPL, tag, const)
                n_ms = (new.get("bench") or {}).get("median_ms")
                acc = compare_case(
                    op, args.outdir, case_id, digest_n=8,
                    new_tag=tag, ref_tag="base",
                )
                checks = acc.get("checks", [])
                max_err = max(
                    [c.get("max_abs_err", 0.0) for c in checks if "max_abs_err" in c],
                    default=None,
                )
                rows.append(dict(
                    op=op, case=case_id, variant=tag, new_const=const,
                    accuracy=acc.get("status", "n/a"),
                    max_abs_err=max_err,
                    new_ms=n_ms, ref_ms=ref_ms,
                    speedup=(ref_ms / n_ms) if (n_ms and ref_ms) else None,
                    new_status=new["status"], ref_status=ref["status"],
                    bench_mode=(new.get("bench") or {}).get("mode"),
                    ref_mode=ref_mode,
                    graph_capture_failed=(new.get("bench") or {}).get(
                        "graph_capture_failed"),
                    notes=new.get("notes"),
                ))
                mark = "ok " if acc.get("status") == "ok" else "BAD"
                _log(
                    f"    {tag} {const:<34} {mark} "
                    f"new={n_ms if n_ms else '-'} ms  ref={ref_ms if ref_ms else '-'} ms"
                )
                if not args.keep_variant_artifacts:
                    try:
                        os.remove(
                            _art_path(args.outdir, op, case_id, NEW_IMPL, tag)
                        )
                    except OSError:
                        pass
            # ref@base is only needed while comparing this case's variants
            if not args.keep_variant_artifacts:
                try:
                    os.remove(_art_path(args.outdir, op, case_id, REF_IMPL, "base"))
                except OSError:
                    pass
    _json_dump(os.path.join(args.outdir, "sweep.json"), rows)
    md = ["# Sweep", "", f"`--sweep {args.sweep}`  measure={args.measure_mode}", ""]
    md.append("`new@vN` and `ref@base` are always measured in the same mode; "
              "a `ref` that cannot be graph-captured drops the whole case to "
              "`loop` timing.")
    md.append("| op | case | variant | const | acc | max_abs_err | new (ms) | ref (ms) "
              "| ref/new | mode |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = lambda v, sp="{:.3f}": sp.format(v) if isinstance(v, (int, float)) else "-"
        md.append("| {op} | {case} | {variant} | `{new_const}` | {acc} | {err} | {n} | {r} | {s} |".format(
            op=r["op"], case=r["case"], variant=r["variant"], new_const=r["new_const"],
            acc=r["accuracy"], err=f(r["max_abs_err"], "{:.3g}"), n=f(r["new_ms"]),
            r=f(r["ref_ms"]), s=f(r["speedup"], "{:.2f}x"), m=r.get("bench_mode")))
    # best accuracy-preserving variant per case
    md += ["", "## Best accuracy-preserving variant per case", ""]
    md.append("| op | case | best variant | const | new (ms) | ref (ms) | ref/new |")
    md.append("|---|---|---|---|---|---|---|")
    for op in ops:
        for case_id in cases[op]:
            good = [r for r in rows
                    if r["op"] == op and r["case"] == case_id
                    and r["accuracy"] == "ok" and r["new_ms"]]
            if not good:
                md.append(f"| {op} | {case_id} | - | no passing variant | - | - | - |")
                continue
            b = min(good, key=lambda r: r["new_ms"])
            f = lambda v, sp="{:.3f}": sp.format(v) if isinstance(v, (int, float)) else "-"
            md.append("| {op} | {case} | {v} | `{c}` | {n} | {r} | {s} |".format(
                op=op, case=case_id, v=b["variant"], c=b["new_const"],
                n=f(b["new_ms"]), r=f(b["ref_ms"]), s=f(b["speedup"], "{:.2f}x")))
    path = os.path.join(args.outdir, "sweep.md")
    with open(path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    _log(f"  -> {path}")
    return rows


def compare(outdir, ops, cases, digest_n=16):
    _log("=" * 78)
    _log("COMPARE  (cpu only)")
    _log("=" * 78)
    out = []
    digest_dir = os.path.join(outdir, "digest")
    for op in ops:
        for case_id in cases[op]:
            res = compare_case(op, outdir, case_id, digest_n=digest_n)
            out.append(res)
            if res["status"] == "skipped":
                _log(f"  {op}/{case_id}: SKIP ({res.get('reason')})")
                continue
            if res["status"] == "error":
                _log(f"  {op}/{case_id}: ERROR ({res.get('reason')})")
                continue
            parts = []
            for c in res["checks"]:
                if c.get("error"):
                    parts.append(f"{c['tensor']}:ERR({c['error']})")
                elif "max_abs_err" in c:
                    parts.append(f"{c['tensor']}:{'ok' if c['ok'] else 'DIFF'}@{c['max_abs_err']:.3g}")
                else:
                    extra = ""
                    if c.get("all_rows_trivial"):
                        extra = f" (degenerate: all {c['rows']} rows are arange)"
                    parts.append(f"{c['tensor']}:{'ok' if c['ok'] else 'DIFF'}{extra}")
            _log(f"  {op}/{case_id}: {res['status']:<9} " + "  ".join(parts))
            # a few-hundred-byte replacement for the raw tensors on failures
            if res["status"] != "ok":
                os.makedirs(digest_dir, exist_ok=True)
                _json_dump(
                    os.path.join(digest_dir, f"{op}__{case_id}.json"),
                    dict(
                        op=op,
                        case=case_id,
                        status=res["status"],
                        rtol=TOLERANCE.get(op, (None, None))[0],
                        atol=TOLERANCE.get(op, (None, None))[1],
                        # note: flat_index is a row-major index into the
                        # compared tensor, whose shape is in `checks[].shape`
                        checks=[
                            c for c in res["checks"] if not c.get("ok")
                        ],
                    ),
                )
    _json_dump(os.path.join(outdir, "accuracy.json"), out)
    return out


def _dir_mb(path):
    total = 0
    for dirpath, _d, filenames in os.walk(path):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return total / 2**20


def apply_artifacts_policy(outdir, ops, cases, policy, accuracy):
    """Shrink the output folder so that only what you need gets downloaded.

    ``all``      keep every .pt (allows a full offline re-compare)
    ``failures`` keep only the cases that did not come back ``ok``
    ``drop``     keep none (accuracy is already in accuracy.json)
    """
    adir = os.path.join(outdir, "artifacts")
    if not os.path.isdir(adir) or policy == "all":
        return
    status = {(r["op"], r["case"]): r.get("status") for r in accuracy or []}
    removed = kept = 0
    freed = 0.0
    for op in ops:
        for case_id in cases[op]:
            # A lone artifact is useless for `compare` -- it needs both sides.
            both = all(
                os.path.isfile(_art_path(outdir, op, case_id, i)) for i in IMPLS
            )
            keep = (
                policy == "failures"
                and both
                and status.get((op, case_id)) != "ok"
            )
            for impl in IMPLS:
                path = _art_path(outdir, op, case_id, impl)
                if not os.path.isfile(path):
                    continue
                if keep:
                    kept += 1
                    continue
                try:
                    freed += os.path.getsize(path) / 2**20
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
    _log(
        f"  artifacts policy={policy}: kept {kept}, removed {removed} "
        f"({freed:.0f} MB freed)"
    )


# =========================================================================== #
# Report
# =========================================================================== #


def _ir_manifest(outdir):
    root = os.path.join(outdir, "ir")
    rows = []
    if not os.path.isdir(root):
        return rows
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            parts = rel.split(os.sep)
            op = parts[0] if len(parts) > 1 else "?"
            impl = parts[1] if len(parts) > 2 else "?"
            ext = fn.rsplit(".", 1)[-1]
            rows.append(
                dict(
                    op=op,
                    impl=impl,
                    kernel=fn.rsplit(".", 1)[0],
                    ext=ext,
                    bytes=os.path.getsize(full),
                    path=rel,
                )
            )
    return rows


def _ir_bundle(outdir):
    """Concatenate per-(op,impl) text IR into one file for easy diffing."""
    root = os.path.join(outdir, "ir")
    bundle = os.path.join(outdir, "ir_bundle")
    if not os.path.isdir(root):
        return
    os.makedirs(bundle, exist_ok=True)
    wanted = ("ttir", "ttadapter")
    groups = {}
    for dirpath, _d, filenames in os.walk(root):
        for fn in filenames:
            ext = fn.rsplit(".", 1)[-1]
            if ext not in wanted:
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            parts = rel.split(os.sep)
            if len(parts) < 3:
                continue
            key = (parts[0], parts[1], ext)
            groups.setdefault(key, []).append(os.path.join(dirpath, fn))
    for (op, impl, ext), files in sorted(groups.items()):
        dst = os.path.join(bundle, f"{op}__{impl}.{ext}.mlir")
        with open(dst, "w") as f:
            for path in sorted(files):
                f.write(f"\n// ===== {os.path.relpath(path, root)} =====\n")
                with open(path, errors="replace") as src:
                    f.write(src.read())
        _log(f"  ir bundle: {os.path.relpath(dst, outdir)}")


def _fake_artifact(op, perturb=False):
    """Small synthetic artifact with the exact keys collect_*() produces."""
    import torch

    torch.manual_seed(11)
    if op == "causal_conv1d_fn":
        # collector layout: out is compacted to the live sequences only
        dim, seqlens, lives = 8, [3, 5, 2, 4], [True, True, False, True]
        live_cols = sum(n for n, l in zip(seqlens, lives) if l)
        return dict(
            out=torch.randn(dim, live_cols, dtype=torch.bfloat16),
            conv_states=torch.randn(len(seqlens), dim, 3),
            lives=lives,
            seqlens=seqlens,
        )
    if op == "compressor":
        return dict(
            out=torch.randn(12, 8, dtype=torch.bfloat16),
            state_cache=torch.randn(2, 4, 8),
            valid_rows=[6, 6],
            rank3=True,
        )
    if op == "gqa":
        return dict(out=torch.randn(2, 16, 4, 8, dtype=torch.bfloat16))
    if op == "kv_rmsnorm_rope_cache":
        return dict(
            k_rope=torch.randn(6, 64, dtype=torch.bfloat16),
            c_kv=torch.randn(6, 512, dtype=torch.bfloat16),
            k_cache=torch.randn(6, 64, dtype=torch.bfloat16),
            ckv_cache=torch.randn(6, 512, dtype=torch.bfloat16),
        )
    if op == "lightning_indexer":
        idx = torch.full((4, 1, 8), -1, dtype=torch.int32)
        idx[0, 0, :0] = torch.tensor([], dtype=torch.int32)
        idx[1, 0, :3] = torch.tensor([7, 0, 3])
        idx[2, 0, :5] = torch.tensor([4, 1, 2, 6, 5])
        idx[3, 0, :8] = torch.arange(8)
        return dict(indices=idx)
    if op == "sparse_attn_sharedkv":
        return dict(
            out=torch.randn(4, 64, 512, dtype=torch.bfloat16),
            lse=torch.empty(0),
        )
    raise KeyError(op)


def selfcheck(outdir=None):
    """Validate the harness itself (builders + comparison) without an NPU.

    Run this once on the target machine before trusting a benchmark run.
    """
    import tempfile

    import torch

    outdir = outdir or tempfile.mkdtemp(prefix="commonir_selfcheck_")
    os.makedirs(os.path.join(outdir, "artifacts"), exist_ok=True)
    failures = []

    def _emit(op, case, impl, art):
        torch.save(art, _art_path(outdir, op, case, impl))

    def _expand_cc1d(art):
        """Reference-side layout: one output column per input token."""
        out = torch.zeros(
            art["out"].shape[0], sum(art["seqlens"]), dtype=art["out"].dtype
        )
        pos = cur = 0
        for n, live in zip(art["seqlens"], art["lives"]):
            if live:
                out[:, pos : pos + n] = art["out"][:, cur : cur + n]
                cur += n
            pos += n
        return dict(art, out=out)

    for op in OP_ORDER:
        base = _fake_artifact(op)
        _emit(op, "same", NEW_IMPL, base)
        ref_art = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
        if op == "causal_conv1d_fn":
            # the counterpart keeps the pad columns -> different column count
            ref_art = _expand_cc1d(base)
        _emit(op, "same", REF_IMPL, ref_art)
        res = compare_case(op, outdir, "same")
        ok = res.get("status") == "ok"
        failures += [] if ok else [f"{op}: identical artifacts not reported as ok"]
        print(f"  {'PASS' if ok else 'FAIL'}  {op:<24} identical -> {res.get('status')}")

        # perturb the largest float tensor and expect a MISMATCH
        key = next(
            (
                k
                for k, v in base.items()
                if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 1
            ),
            None,
        )
        bad = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
        if key is None:
            idx = bad["indices"].clone()
            idx[3, 0, 0] = 99
            bad["indices"] = idx
        else:
            bad[key] = bad[key] + 1.0
        _emit(op, "perturbed", NEW_IMPL, base)
        _emit(op, "perturbed", REF_IMPL, bad)
        res = compare_case(op, outdir, "perturbed")
        ok = res.get("status") == "MISMATCH"
        failures += [] if ok else [f"{op}: perturbation not detected"]
        print(f"  {'PASS' if ok else 'FAIL'}  {op:<24} perturbed -> {res.get('status')}")

    # lightning_indexer output order is free -> a permutation must still pass
    a = _fake_artifact("lightning_indexer")
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in a.items()}
    row = b["indices"][2, 0, :5].clone()
    b["indices"][2, 0, :5] = row.flip(0)
    _emit("lightning_indexer", "perm", NEW_IMPL, a)
    _emit("lightning_indexer", "perm", REF_IMPL, b)
    res = compare_case("lightning_indexer", outdir, "perm")
    ok = res.get("status") == "ok"
    failures += [] if ok else ["lightning_indexer: index permutation rejected"]
    print(f"  {'PASS' if ok else 'FAIL'}  {'lightning_indexer':<24} permuted idx -> {ok}")

    # causal_conv1d_fn: the ref keeps pad columns, the new op compacts them
    a = _fake_artifact("causal_conv1d_fn")
    _emit("causal_conv1d_fn", "pad", NEW_IMPL, a)
    _emit("causal_conv1d_fn", "pad", REF_IMPL, _expand_cc1d(a))
    res = compare_case("causal_conv1d_fn", outdir, "pad")
    ok = res.get("status") == "ok"
    failures += [] if ok else ["causal_conv1d_fn: pad-slot compaction not handled"]
    print(f"  {'PASS' if ok else 'FAIL'}  {'causal_conv1d_fn':<24} pad compaction -> {ok}")

    # collectors: a NameError/typo in a collect_*() silently kills the whole op
    # for every case (this actually happened), so exercise each one here.
    fake = {
        "causal_conv1d_fn": (
            dict(conv_states=torch.randn(4, 8, 3)),
            dict(slots=[0, 1], live=[True, False], seqlens=[3, 2]),
            torch.randn(8, 6),
            ("out", "conv_states"),
        ),
        "compressor": (
            dict(state_cache=torch.randn(2, 4, 8)),
            dict(valid_rows=[3], rank3=False),
            torch.randn(5, 8),
            ("out", "state_cache"),
        ),
        "gqa": (dict(), dict(), torch.randn(1, 4, 2, 8), ("out",)),
        "kv_rmsnorm_rope_cache": (
            dict(
                k_cache=torch.zeros(4, 2, 1, 8),
                ckv_cache=torch.zeros(4, 2, 1, 16),
                index=torch.arange(6),
            ),
            dict(rope=8, rms=16),
            (None, None, torch.randn(6, 8), torch.randn(6, 16)),
            ("k_rope", "c_kv", "k_cache", "ckv_cache"),
        ),
        "lightning_indexer": (
            dict(),
            dict(),
            (torch.zeros((2, 1, 4), dtype=torch.int32), torch.empty(0)),
            ("indices",),
        ),
        "sparse_attn_sharedkv": (
            dict(),
            dict(),
            (torch.randn(2, 4, 8), torch.empty(0)),
            ("out", "lse"),
        ),
    }
    for op, (kw, meta, result, keys) in fake.items():
        try:
            got = COLLECTORS[op](kw, meta, result)
            missing = [k for k in keys if k not in got]
            ok = not missing
            detail = f"keys ok" if ok else f"missing {missing}"
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        failures += [] if ok else [f"{op}: collector broken ({detail})"]
        print(f"  {'PASS' if ok else 'FAIL'}  {op:<24} collector -> {detail}")

    # the gqa torch fallback golden must agree with a naive bottom-right loop
    def _naive_gqa(q, k, v, scale, causal):
        B, S, Hq, D = q.shape
        Skv, Hkv = k.shape[1], k.shape[2]
        g = Hq // Hkv
        o = torch.zeros_like(q, dtype=torch.float32)
        for b in range(B):
            for h in range(Hq):
                kh = h // g
                s = (q[b, :, h].float() @ k[b, :, kh].float().T) * scale
                if causal:
                    i = torch.arange(S)[:, None]
                    j = torch.arange(Skv)[None, :]
                    s = s.masked_fill(j > i + (Skv - S), float("-inf"))
                o[b, :, h] = torch.softmax(s, -1) @ v[b, :, kh].float()
        return o

    for (S, Skv, Hq, Hkv, D, causal) in ((32, 32, 8, 4, 16, True),
                                         (16, 48, 8, 2, 16, True),
                                         (16, 16, 4, 1, 16, False)):
        q = torch.randn(1, S, Hq, D, dtype=torch.bfloat16)
        k = torch.randn(1, Skv, Hkv, D, dtype=torch.bfloat16)
        v = torch.randn(1, Skv, Hkv, D, dtype=torch.bfloat16)
        scale = D ** -0.5
        kw = dict(query=q, key=k, value=v, scaleValue=-1.0, is_causal=causal)
        try:
            got = _gqa_torch_golden(kw).float()
            ref = _naive_gqa(q, k, v, scale, causal)
            delta = float((got - ref).abs().max())
            ok = delta < 0.02
            detail = f"S{S}x{Skv} causal={causal} max|d|={delta:.4f}"
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        failures += [] if ok else [f"gqa golden disagrees with naive: {detail}"]
        print(f"  {'PASS' if ok else 'FAIL'}  {'gqa torch golden':<24} -> {detail}")

    print()
    if failures:
        for f in failures:
            print(f"  FAILURE: {f}")
        return 1
    print(f"selfcheck OK  (scratch dir: {outdir})")
    return 0


def report(outdir):
    _log("=" * 78)
    _log("REPORT")
    _log("=" * 78)

    produce = _json_load(os.path.join(outdir, "produce.json"), []) or []
    accuracy = _json_load(os.path.join(outdir, "accuracy.json"), []) or []

    def key(r):
        return (r["op"], r["case"], r["impl"])

    timings = {key(r): r for r in produce}
    acc = {(r["op"], r["case"]): r for r in accuracy}

    ops = [op for op in OP_ORDER if any(r["op"] == op for r in produce)] or list(
        OP_ORDER
    )
    case_order = []
    for r in produce:
        pair = (r["op"], r["case"])
        if pair not in case_order:
            case_order.append(pair)

    rows = []
    for op, case_id in case_order:
        new = timings.get((op, case_id, NEW_IMPL), {})
        ref = timings.get((op, case_id, REF_IMPL), {})
        a = acc.get((op, case_id), {})
        n_ms, r_ms = new.get("median_ms"), ref.get("median_ms")
        n_mode, r_mode = new.get("measure_mode"), ref.get("measure_mode")
        mode_ok = n_mode == r_mode and n_mode in (
            "loop", "graph", "do_bench", "host-split")
        speedup = (r_ms / n_ms) if (n_ms and r_ms and mode_ok) else None
        checks = a.get("checks", [])
        verdict = a.get("status", "n/a")
        max_err = max(
            [c.get("max_abs_err", 0.0) for c in checks if "max_abs_err" in c],
            default=None,
        )
        # how many elements actually got compared -- guards against a
        # "passing" case whose comparison was vacuous (0-element slice)
        compared = sum(int(c.get("numel", 0)) for c in checks if "numel" in c)
        rows.append(
            dict(
                op=op,
                case=case_id,
                accuracy=verdict,
                compared_elems=compared,
                max_abs_err=max_err,
                new_status=new.get("status", "n/a"),
                ref_status=ref.get("status", "n/a"),
                new_ms=n_ms,
                ref_ms=r_ms,
                speedup=speedup,
                new_mode=n_mode,
                ref_mode=r_mode,
                mode_comparable=mode_ok,
                new_host_ms=new.get("host_ms"),
                new_device_ms=new.get("device_ms"),
                new_peak_alloc_mb=new.get("peak_alloc_mb"),
                new_artifact_mb=new.get("artifact_mb"),
                ref_artifact_mb=ref.get("artifact_mb"),
            )
        )

    # ---- json -----------------------------------------------------------
    _json_dump(os.path.join(outdir, "results.json"), rows)

    # ---- csv ------------------------------------------------------------
    csv_path = os.path.join(outdir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["op"])
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})

    ir = _ir_manifest(outdir)
    if ir:
        _json_dump(os.path.join(outdir, "ir_manifest.json"), ir)
        _ir_bundle(outdir)

    # ---- markdown -------------------------------------------------------
    md = []
    md.append("# CommonIR (TLE/DSA) rewrite vs non-CommonIR counterpart\n")
    _run_meta = _json_load(os.path.join(outdir, "meta.json"), {}) or {}
    md.append(
        "- device: `{}`".format(
            _run_meta.get("device")
            or os.environ.get("COMMONIR_BENCH_DEVICE", "?")
        )
    )
    md.append(f"- output: `{os.path.abspath(outdir)}`")
    md.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Accuracy + latency\n")
    md.append(
        "| op | case | accuracy | compared | max_abs_err | new (ms) | ref (ms) | ref/new |"
    )
    md.append("|---|---|---|---|---|---|---|---|")

    def fmt(v, spec="{:.3f}"):
        return spec.format(v) if isinstance(v, (int, float)) else "-"

    for row in rows:
        md.append(
            "| {op} | {case} | {acc} | {cmp} | {err} | {n} | {r} | {sp} |".format(
                op=row["op"],
                case=row["case"],
                acc=row["accuracy"],
                cmp=fmt(row.get("compared_elems"), "{:.3g}") if row.get("compared_elems") else "0 !",
                err=fmt(row["max_abs_err"], "{:.3g}"),
                n=fmt(row["new_ms"]),
                r=fmt(row["ref_ms"]),
                sp=(fmt(row["speedup"], "{:.2f}x")
                    if row.get("mode_comparable") else "mode!"),
            )
        )

    missing = [
        (r["op"], r["case"], r["impl"], r["status"])
        for r in produce
        if r["status"] != "ok"
    ]
    if missing:
        md.append("\n## Failed / skipped runs\n")
        md.append("| op | case | impl | status |")
        md.append("|---|---|---|---|")
        for op, case_id, impl, status in missing:
            md.append(f"| {op} | {case_id} | {impl} | {status} |")

    per_op = {}
    for row in rows:
        if row["speedup"]:
            per_op.setdefault(row["op"], []).append(row["speedup"])
    if per_op:
        md.append("\n## Per-op geomean speedup (ref/new)\n")
        md.append("| op | geomean | cases |")
        md.append("|---|---|---|")
        for op in sorted(per_op):
            vals = per_op[op]
            g = math.exp(sum(math.log(v) for v in vals) / len(vals))
            md.append(f"| {op} | {g:.2f}x | {len(vals)} |")

    # ---- workarounds applied to the counterpart, or kernel overrides --------
    seen = {}
    for r in produce:
        for n in r.get("notes") or []:
            seen.setdefault((r["impl"], n), f"{r['op']}/{r['case']}")
    applied = (_json_load(os.path.join(outdir, "meta.json"), {}) or {}).get(
        "new_const"
    )
    if seen or applied:
        md.append("\n## Workarounds applied during this run\n")
        for (impl, note), sample in sorted(seen.items()):
            md.append(f"- `{impl}`: {note} (first seen in `{sample}`)")
        if applied:
            md.append(
                "- `new`: module constants overridden, **these are not the "
                f"committed kernel's numbers**: `{applied}`"
            )

    md.append("\n## Counterparts\n")
    for op in ops:
        md.append(f"- **{op}**: `{REF_SOURCE[op]}`")

    if ir:
        md.append("\n## IR dump\n")
        md.append(f"{len(ir)} files under `ir/<op>/<impl>/`.")
        md.append("`ir_bundle/<op>__<impl>.{ttir,ttadapter}.mlir` holds the concatenated text IR.")
        md.append("`ir_manifest.json` has the full listing.")

    # ---- what does this folder actually cost to move? --------------------
    sizes = {
        name: _dir_mb(os.path.join(outdir, name))
        for name in ("artifacts", "ir", "logs", "raw", "digest")
    }
    top = sum(
        os.path.getsize(os.path.join(outdir, f)) / 2**20
        for f in (
            "accuracy.json",
            "results.json",
            "results.csv",
            "produce.json",
            "meta.json",
            "ir_manifest.json",
            "summary.md",
        )
        if os.path.isfile(os.path.join(outdir, f))
    )
    md.append("\n## Output size (MB)\n")
    md.append("| subfolder | MB | needed for |")
    md.append("|---|---|---|")
    md.append(f"| top-level .json/.csv/.md | {top:.2f} | the result table |")
    md.append(
        f"| `artifacts/` | {sizes['artifacts']:.2f} | only re-running `compare` offline |"
    )
    md.append(f"| `ir/` | {sizes['ir']:.2f} | the IR analysis |")
    md.append(f"| `digest/` | {sizes['digest']:.2f} | worst-element locations of failures |")
    md.append(f"| `raw/` | {sizes['raw']:.2f} | timing samples + worker env |")
    md.append(f"| `logs/` | {sizes['logs']:.2f} | tracebacks / pass IR |")
    total_mb = sum(sizes.values()) + top
    md.append(f"\n**total {total_mb:.2f} MB**")
    if sizes["artifacts"] >= 50:
        md.append(
            f"\n`artifacts/` is {sizes['artifacts']:.0f} MB of the {total_mb:.0f} MB. "
            "It is only needed to re-run `compare` on another machine; accuracy "
            "has already been computed here. Re-run with `--artifacts drop` (or "
            "`--artifacts failures`, the default) to keep the folder small."
        )
    md.append("\nDownload without the raw tensors:\n")
    md.append("```bash")
    md.append(
        f"tar Jcf run.tar.xz -C {os.path.dirname(os.path.abspath(outdir))} \\"
    )
    md.append(f"    --exclude='{os.path.basename(outdir)}/artifacts' \\")
    md.append(f"    {os.path.basename(outdir)}")
    md.append("```")

    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    _log("")
    _log("\n".join(md[:60]))
    _log("")
    _log(f"  -> {os.path.join(outdir, 'summary.md')}")
    _log(f"  -> {csv_path}")
    _log(f"  -> {os.path.join(outdir, 'results.json')}")


# =========================================================================== #
# Entry point
# =========================================================================== #


def _parse_ops(value):
    if not value or value == "all":
        return list(OP_ORDER)
    ops = [o.strip() for o in value.split(",") if o.strip()]
    unknown = [o for o in ops if o not in OP_ORDER]
    if unknown:
        raise SystemExit(f"unknown op(s): {unknown}\navailable: {list(OP_ORDER)}")
    return ops


def _add_common(p):
    p.add_argument(
        "--outdir",
        default=os.path.join(_HERE, "out", time.strftime("%Y%m%d-%H%M%S")),
        help="output folder (created if missing)",
    )
    p.add_argument("--ops", default="all", help="comma separated, or 'all'")
    p.add_argument(
        "--cases",
        default="quick",
        help="quick|default|full, or a comma separated case-id / set-name list",
    )
    p.add_argument("--kernel-dir", default=_KERNEL_DIR_DEFAULT)
    p.add_argument("--device", default=None, help="npu | npu:1 | cpu (default: auto)")


def _add_produce_args(p):
    p.add_argument("--dump-ir", action="store_true", help="TRITON_KERNEL_DUMP=1")
    p.add_argument(
        "--dump-pass-ir",
        default=None,
        metavar="PASS",
        help="MLIR_ENABLE_DUMP=<PASS> (1 = all passes); very verbose, goes to logs/",
    )
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument(
        "--measure-mode",
        choices=("loop", "do_bench", "npu-graph", "host-split"),
        default="loop",
        help="loop = wall clock (host+device); npu-graph = device only via "
        "graph replay; host-split = wall clock + launch-stubbed host cost",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument(
        "--sparse-metadata",
        choices=("auto", "zeros"),
        default="auto",
        help="sparse_attn_sharedkv reference metadata source",
    )
    p.add_argument(
        "--new-const",
        default=None,
        metavar="NAME=VALUE[,NAME=VALUE]",
        help="experiment only: override module-level constants of the CommonIR "
        "kernel before calling (e.g. '_BLOCK_N=64' for gqa to test whether a "
        "UB overflow is just a too-large tile). Recorded in meta.notes.",
    )


def _add_artifact_arg(p):
    p.add_argument(
        "--save-artifacts",
        dest="save_artifacts",
        action="store_true",
        default=True,
        help="save per-impl CPU artifacts (needed by the CPU-only compare stage)",
    )
    p.add_argument(
        "--no-save-artifacts",
        dest="save_artifacts",
        action="store_false",
        help="perf-only run: no artifacts AND no accuracy stage",
    )
    p.add_argument(
        "--artifacts",
        dest="artifacts_policy",
        choices=("all", "failures", "drop"),
        default="failures",
        help="what to keep after compare: 'all' = every .pt (offline re-compare), "
        "'failures' (default) = only the cases that did not pass, "
        "'drop' = none (accuracy is already in accuracy.json)",
    )
    p.add_argument(
        "--digest-n",
        type=int,
        default=16,
        help="on a mismatch, record the N worst element coordinates+values "
        "(default 16; 0 disables)",
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="commonir_bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="produce -> compare -> report")
    _add_common(p_run)
    _add_produce_args(p_run)
    _add_artifact_arg(p_run)
    p_run.add_argument(
        "--reuse",
        action="store_true",
        help="keep existing raw/*.json and artifacts instead of re-running",
    )

    p_prod = sub.add_parser("produce", help="run the workers only")
    _add_common(p_prod)
    _add_produce_args(p_prod)
    _add_artifact_arg(p_prod)

    p_cmp = sub.add_parser("compare", help="CPU-only accuracy comparison")
    _add_common(p_cmp)
    p_cmp.add_argument(
        "--digest-n",
        type=int,
        default=16,
        help="on a mismatch, record the N worst element coordinates+values",
    )

    p_rep = sub.add_parser("report", help="aggregate raw results")
    _add_common(p_rep)

    p_list = sub.add_parser("list", help="print the op/case registry and exit")
    _add_common(p_list)

    p_sw = sub.add_parser(
        "sweep",
        help="time the CommonIR kernel over a grid of --new-const values, "
        "accuracy-checked against the counterpart",
    )
    _add_common(p_sw)
    _add_produce_args(p_sw)
    _add_artifact_arg(p_sw)
    p_sw.add_argument(
        "--sweep",
        required=True,
        metavar="NAME=V1|V2,...",
        help="comma separated knobs, '|' separated values: "
        "'_PROJ_BK=128|256|512,_PROJ_BN=128|256'",
    )
    p_sw.add_argument(
        "--keep-variant-artifacts",
        action="store_true",
        help="keep every variant's .pt (default: delete after comparing)",
    )

    p_chk = sub.add_parser(
        "selfcheck", help="validate this harness itself (no NPU required)"
    )
    p_chk.add_argument("--outdir", default=None)

    p_w = sub.add_parser("worker", help="single (op, case, impl) run")
    _add_common(p_w)
    _add_produce_args(p_w)
    _add_artifact_arg(p_w)
    p_w.add_argument("--op", required=True, choices=list(OP_ORDER))
    p_w.add_argument("--case", required=True)
    p_w.add_argument("--impl", required=True, choices=list(IMPLS))
    p_w.add_argument(
        "--impl-tag", default="", help="suffix for raw/, logs/, artifacts/ (sweeps)"
    )

    args = ap.parse_args(argv)

    if args.cmd == "selfcheck":
        os.environ["COMMONIR_BENCH_DEVICE"] = "cpu"
        return selfcheck(args.outdir)

    if args.device is None:
        args.device = _detect_device()
    os.environ["COMMONIR_BENCH_DEVICE"] = str(args.device)

    if args.cmd == "worker":
        return worker_main(args)

    ops = _parse_ops(args.ops)
    cases = _resolve_cases(ops, getattr(args, "cases", "quick"), getattr(args, "cases", None))
    os.makedirs(args.outdir, exist_ok=True)

    if args.cmd == "run":
        meta = dict(
            outdir=os.path.abspath(args.outdir),
            ops=ops,
            cases={k: list(v) for k, v in cases.items()},
            device=args.device,
            kernel_dir=os.path.abspath(args.kernel_dir),
            dump_ir=args.dump_ir,
            dump_pass_ir=args.dump_pass_ir,
            measure_mode=args.measure_mode,
            new_const=getattr(args, "new_const", None),
            env={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        )
        _json_dump(os.path.join(args.outdir, "meta.json"), meta)
        if not args.reuse:
            produce(args.outdir, ops, cases, args)
        else:
            _log("--reuse: skipping produce stage")
        accuracy = None
        if args.save_artifacts:
            accuracy = compare(
                args.outdir, ops, cases, digest_n=args.digest_n
            )
            apply_artifacts_policy(
                args.outdir, ops, cases, args.artifacts_policy, accuracy
            )
        else:
            _log("--no-save-artifacts: skipping compare stage (perf-only run)")
        report(args.outdir)
        return 0

    if args.cmd == "produce":
        produce(args.outdir, ops, cases, args)
        return 0

    if args.cmd == "sweep":
        sweep(args, ops, cases)
        return 0

    if args.cmd == "compare":
        # default to the case set that was actually produced
        produced = (_json_load(os.path.join(args.outdir, "meta.json"), {}) or {}).get(
            "cases"
        )
        if produced and args.cases in ("quick", "default", "full"):
            cases = {
                op: list(produced.get(op, cases[op])) for op in ops
            }
        compare(args.outdir, ops, cases, digest_n=args.digest_n)
        return 0

    if args.cmd == "report":
        report(args.outdir)
        return 0

    if args.cmd == "list":
        for op in ops:
            print(f"\n{op}")
            print(f"  counterpart: {REF_SOURCE[op]}")
            print(f"  tolerance  : {TOLERANCE.get(op, 'index multiset')}")
            print(f"  compare    : {COMPARE_MODE[op]}")
            print(f"  source     : {os.path.join(os.path.abspath(args.kernel_dir), op + '.py')}")
            ids = _case_ids(op)
            for set_name in ("quick", "default", "full", "heavy"):
                print(f"    {set_name:<8}: {', '.join(CASES[op][set_name]) or '-'}")
            print(f"    all ids  : {', '.join(sorted(ids))}")
        return 0

    raise SystemExit(f"unknown command {args.cmd}")


def _detect_device():
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


if __name__ == "__main__":
    sys.exit(main())
