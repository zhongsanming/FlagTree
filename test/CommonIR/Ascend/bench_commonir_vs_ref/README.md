# commonir_bench — CommonIR(TLE/DSA) rewrite vs. its non-CommonIR counterpart

Accuracy / performance / IR harness for the six kernels added in
`test/CommonIR/Ascend/success_case/` (commit *"add kernelgen ops"*), each
compared against the functionally-equivalent implementation that does **not**
use the `tle.dsa.*` (CommonIR) primitives.

Everything runs on the NPU box; the produced folder can then be copied back and
re-analysed on a machine **without** an accelerator, because the accuracy stage
is pure CPU.

---

## 1. Counterpart mapping

| # | CommonIR kernel (`success_case/`) | counterpart (no `tle.dsa`) | upstream repo |
|---|---|---|---|
| 1 | `causal_conv1d_fn.py` | `src/flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_fn.py` → `causal_conv1d_fn` | FlagGems-vllm |
| 2 | `compressor.py` | `src/flaggems_vllm/runtime/backend/_ascend/ops/compressor.py` → `compressor` | FlagGems-vllm |
| 3 | `gqa.py` | `src/flaggems_vllm/ops/attention.py` → `flash_attention_forward` (→ `flaggems/.../flash_api.py::mha_fwd`, `flash_kernel.py::_attn_fwd`) | FlagGems-vllm / FlagGems |
| 4 | `kv_rmsnorm_rope_cache.py` | `src/flaggems_vllm/runtime/backend/_ascend/ops/kv_rmsnorm_rope_cache.py` → `kv_rmsnorm_rope_cache` | FlagGems-vllm |
| 5 | `lightning_indexer.py` | `src/flaggems_vllm/runtime/backend/_ascend/ops/lightning_indexer.py` → `lightning_indexer` | FlagGems-vllm |
| 6 | `sparse_attn_sharedkv.py` | `src/flaggems_vllm/runtime/backend/_ascend/ops/sparse_attn_sharedkv.py` → `sparse_attn_sharedkv` | FlagGems-vllm |

Notes that matter for interpreting results:

* **#3 `gqa` has no same-named op anywhere.** `run(query, key, value, scaleValue,
  is_causal)` over `[B,S,H,D]` is the FlashAttention / SDPA contract. The
  equivalence is not approximate:
  `flash_kernel.py` computes `n_block_max = cdiv((m_block+1)*BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)`
  which is exactly `gqa.py`'s `tl.cdiv((pid_m+1)*BLOCK_M + (S_KV - S), BLOCK_N)`,
  and `k_ptr += hid // h_hk_ratio` is exactly `hkv = hq // (N_q // N_kv)`.
  The adapter in this harness transposes BSND→BHSD and back.
* **#4 and #5 still use some `tle.*` primitives** in their counterparts
  (`extract_slice`/`insert_slice`/`parallel`, and `tle.dsa.ascend.raw` sort
  ops respectively) — they just do not use the `dsa.copy/alloc/to_buffer`
  memory primitives. Flagged so the "no-CommonIR" label is not overstated.
* **#6** reference needs a 1024-int32 `metadata` blob. Content is validated but
  never read by the Triton paths, so the harness uses `vllm_ascend`'s metadata
  op when importable and falls back to zeros (`--sparse-metadata zeros`).

---

## 2. Quick start (on the NPU box)

```bash
cd <flagtree>/test/CommonIR/Ascend/bench_commonir_vs_ref

# 0) validate the harness itself first -- no NPU, no triton, ~2 s
python commonir_bench.py selfcheck

# 1) what will run
python commonir_bench.py list

# 2) smoke run: 2 cases per op, with IR dumps
python commonir_bench.py run --outdir out/smoke --cases quick --dump-ir

# 3) the real thing
python commonir_bench.py run --outdir out/full --cases default --dump-ir \
       --iters 20 --rounds 5 --warmup 5
```

Then copy the folder back:

```bash
tar czf commonir_full.tgz out/full
```

On the machine that has the folder but no NPU:

```bash
python commonir_bench.py compare --outdir out/full   # recompute accuracy (CPU)
python commonir_bench.py report  --outdir out/full   # rewrite summary.md/csv
```

---

## 3. Case sets

`--cases` accepts `quick` | `default` | `full` | `heavy`, or an explicit
comma-separated list mixing set names and case ids
(e.g. `--cases quick,decode0_start8195`). Run `list` to see them.

| op | quick | default | full |
|---|---|---|---|
| causal_conv1d_fn | 2 | 4 | 6 |
| compressor | 2 | 5 | 9 |
| gqa | 2 | 4 | 6 |
| kv_rmsnorm_rope_cache | 2 | 3 | 4 |
| lightning_indexer | 3 | 5 | 12 |
| sparse_attn_sharedkv | 3 | 5 | 6 |

* `heavy` = shapes whose *input* alone needs ~2 GB (currently only
  `compressor/decode0_b32`). Opt-in, never part of `full`.
* Case shapes are copied from the counterpart repo's own test suites
  (`tests/compressor_cases.py`, `tests/lightning_indexer_utils.py`,
  `tests/sparse_attn_sharedkv_utils.py`, `tests/test_kv_rmsnorm_rope_cache.py`,
  `tests/test_causal_conv1d_fn.py`) so the two implementations see identical data.
* Inputs are rebuilt deterministically (fixed seeds) in each worker, so the
  `new` and `ref` runs are bit-identical on the input side.

---

## 4. Accuracy semantics (per op)

| op | compared tensors | comparison | tolerance |
|---|---|---|---|
| causal_conv1d_fn | `out`, in-place `conv_states` (touched slots) | tensor | rtol 1e-2 / atol 5e-2 |
| compressor | `out` (first `valid_rows` rows), in-place `state_cache` | tensor | rtol 5e-2 / atol 5e-2 |
| gqa | `out` | tensor | rtol 2e-2 / atol 2e-2 |
| kv_rmsnorm_rope_cache | `k_rope`, `c_kv`, `k_cache`, `ckv_cache` (touched slots) | tensor | rtol 1e-3 / atol 1e-2 |
| lightning_indexer | `indices` | **order-free multiset per row**, `-1` tail ignored | exact |
| sparse_attn_sharedkv | `out`, `lse` (must stay empty) | tensor | rtol 2e-2 / atol 2e-2 |

Tolerances are taken from the counterpart repo's own tests, so "accuracy: ok"
means "as tight as that op is normally validated".

For every tensor the report records `max_abs_err`, `mean_abs_err`,
`max_rel_err`, `p50/p90/p99_abs_err`, `rms_ref`, cosine similarity and NaN
counts — so a "MISMATCH" that is really fp32-accumulation-order noise is easy to
tell apart from a real break. On a mismatch, `digest/<op>__<case>.json`
additionally lists the `--digest-n` (default 16) worst elements as
`{flat_index, new, ref, abs_err}` — row-major flat index into the compared
tensor — which is usually enough to localise the bug without any `.pt` file.
Statistics are streamed in chunks, so a 268M-element `out` does not materialise
a 1 GB float32 copy.

Two interface divergences are handled explicitly rather than papered over:

1. **`causal_conv1d_fn` output width.** The CommonIR op *compacts away*
   `pad_slot_id` sequences (`out` is `[dim, sum(live_len)]`); the counterpart
   keeps one column per input token (`[dim, total_len]`, pad columns
   uninitialised). The comparator detects which layout it received and slices
   the live columns before diffing. Any accuracy problem exclusively in the
   padded case is therefore a real bug, not a shape artefact.
2. **`compressor` only fills the first `valid_rows` output rows** (rows past the
   compressed-group count are left as allocated). Only the valid prefix is
   diffed, matching what the counterpart's own test does.

For every tensor the report records `max_abs_err`, `mean_abs_err`,
`max_rel_err`, `rms_ref`, cosine similarity and NaN counts — so a "MISMATCH"
that is really fp32-accumulation-order noise is easy to tell apart from a real
break.

---

## 5. Performance

* One worker per `(op, case, impl)`; each times only that impl, in its own
  process, on already-warm kernels.
* Default `--measure-mode loop`: `--warmup` calls, then `--rounds` rounds of
  `--iters` calls with a device sync around each round; reports
  `median/min/mean` ms (median is what lands in the table).
* `--measure-mode do_bench` uses `triton.backends.ascend.testing.do_bench_npu`
  when available (`triton.testing.do_bench` otherwise).
* `ref/new` in the summary is the speedup of the CommonIR rewrite.
* `--no-bench` gives an accuracy-only run.

Caveat: some of these ops write to their inputs in place (`conv_states`,
`state_cache`, the paged caches). During the timing loop the same buffers are
re-written every iteration, which is what the real caller does too; the values
converge/drift but the work per launch is unchanged.

---

## 6. IR dumps

`--dump-ir` sets, **per (op, impl) worker**:

```
TRITON_KERNEL_DUMP=1
TRITON_ALWAYS_COMPILE=1          # otherwise the cache hit skips the dump
TRITON_DUMP_DIR=<outdir>/ir/<op>/<impl>
```

which yields every stage the Ascend backend declares (from
`third_party/ascend/backend/compiler.py::add_stages`):

```
<outdir>/ir/<op>/<impl>/<src-hash>/<kernel>.ttir        # Triton IR
<outdir>/ir/<op>/<impl>/<src-hash>/<kernel>.ttadapter   # CommonIR -> Linalg
<outdir>/ir/<op>/<impl>/<src-hash>/<kernel>.source
```

`report` then also writes

* `ir_manifest.json` — every dumped file with op/impl/kernel/ext/size,
* `ir_bundle/<op>__<impl>.{ttir,ttadapter}.mlir` — all kernels of one
  (op, impl) concatenated, which is what you want for `diff`.

For pass-by-pass IR (the full `commonir_to_hivm → ... → triton_to_linalg`
pipeline, which is where the rewrite actually differs) use:

```bash
python commonir_bench.py run --outdir out/pass_ir --cases quick \
       --dump-pass-ir 1          # MLIR_ENABLE_DUMP=1, everything
# or a single pass, e.g.
python commonir_bench.py run --outdir out/pass_ir --cases quick \
       --dump-pass-ir commonir-to-hivm
```

`MLIR_ENABLE_DUMP` writes to stderr, so it lands in
`<outdir>/logs/<op>__<case>__<impl>.log` — these files get very large, which is
why it is not enabled together with `--dump-ir` by default.

---

## 7. Output layout

```
<outdir>/
  meta.json                       # run config + env
  produce.json                    # one row per (op, case, impl) worker
  accuracy.json                   # one row per (op, case) with per-tensor diffs
  results.json / results.csv      # joined table (accuracy + latency + speedup)
  summary.md                      # the human-readable report
  raw/<op>__<case>__<impl>.json   # worker result incl. timing samples
  logs/<op>__<case>__<impl>.log   # worker stdout+stderr (tracebacks, MLIR dumps)
  digest/<op>__<case>.json        # worst-element locations, failures only (~KB)
  artifacts/<op>__<case>__<impl>.pt   # CPU tensors -- see section 8
  ir/...                          # stage IR (only with --dump-ir)
  ir_manifest.json, ir_bundle/    # only if IR was dumped
```

---

## 8. What are the `.pt` files in `artifacts/`? Do I need them?

Each `.pt` is a `torch.save`d dict of **that implementation's output tensors on
CPU** (exactly what the `collect_*()` functions pull out: the op's return value
plus the in-place-updated caches, reduced to the touched rows). Sizes are
1:1 with the real outputs, e.g. an 8192-token `sparse_attn_sharedkv` prefill
`out` is 512 MB per impl, a `compressor` prefill's fp32 `state_cache` ~70 MB.

They exist **only** to feed the CPU-side `compare` stage. Since `run` already
executes `compare` **on the NPU box**, the verdict is in `accuracy.json`, and
the `.pt` files add nothing to it.

| you want… | need the `.pt`? |
|---|---|
| accuracy verdict + error stats | **no** — in `accuracy.json` / `results.csv` / `summary.md` |
| timing + speedup | **no** — in `results.csv` |
| IR diff | **no** — in `ir/`, `ir_bundle/` |
| to see *where* a mismatch is | **no** — in `digest/` (worst-N indices + values) |
| to re-run `compare` on another machine (no NPU) | yes |
| to change tolerances / write your own analysis (error histograms, per-token attribution, bisection) | yes |

Hence the default is `--artifacts failures`: after `compare` the `.pt` of every
**passing** case is deleted, and only failing ones are kept (plus a `digest/`
file for each). Options:

```bash
--artifacts failures   # default: keep only non-ok cases
--artifacts drop       # keep none; accuracy is still computed
--artifacts all        # keep everything (old behaviour; needed for offline re-compare)
--no-save-artifacts    # perf-only: no artifacts AND no accuracy stage
```

`summary.md` ends with an **Output size** table (MB per subfolder, total) plus a
ready-to-paste `tar` that excludes `artifacts/`.

For a full `default` case set (including the 8192-token sparse prefills)
`artifacts/` is a few GB while everything else is a few MB, so in practice:

```bash
python commonir_bench.py run --outdir out/full --cases default --dump-ir
# -> out/full.tar.xz        (MB, everything except artifacts)
# -> out/full.artifacts.tar.xz  (only if something actually failed)
```

---

## 9. Failure handling

Each worker is a subprocess with `--timeout` (default 3600 s):

* uncaught Python exception → `status: error` + full traceback in `logs/`
* segfault / MTE fault / abort → `status: crash` (raw JSON absent)
* hang → `status: timeout`, worker killed, the run continues

A wedged NPU context can still poison later workers in the same process group;
if several consecutive runs crash, re-run with `--ops <single op>` to isolate.

---

## 10. `selfcheck`

`python commonir_bench.py selfcheck` builds synthetic artifact pairs and
verifies that the comparator reports `ok` for identical data, `MISMATCH` for
perturbed data, accepts a permuted `lightning_indexer` index row, and handles
the `causal_conv1d_fn` pad-column compaction. It needs `torch` only. Run it
before trusting any result — it catches harness bugs (and I could not run the
NPU paths while writing this).

---

## 11. Counterpart / toolchain problems this harness works around

Recorded from the first full run (`full-20261020-174740`), so the numbers are
not misread as rewrite regressions:

1. **`mha_fwd` hardcodes `"cuda"`** — `flaggems_vllm/ops/flash_api.py`
   (`dispatch()`, also present in `flag_gems/ops/flash_api.py`) calls
   `torch_device_fn.get_device_properties("cuda")`, which on Ascend raises
   `ValueError: Expected a npu device, but got: cuda`. That makes the `gqa`
   counterpart unusable out of the box. `_patch_flash_api_cuda_literal()`
   retries without the bogus device and supplies a `multi_processor_count`
   stand-in (torch_npu names it differently). The patch is recorded in
   `raw/*.json` under `meta.notes`. **Worth reporting upstream.**

2. **`lightning_indexer` counterpart: bytecode/MLIR skew** — on the
   `top-k` path (`*_prefill_sort_*`, `*_decode_stage1*`) the installed
   `tle/language/dsa/ascend/custom_ops/custom_ops.bc` emits
   `hivm.hir.custom` with `operandSegmentSizes = [0, 0]` but 7 operands, and
   BiShengHIR rejects it (`in bytecode version 6 produced by MLIR22.0.0git`).
   This is a flaggems_vllm↔flagtree version mismatch, not a kernel bug.
   Only the `direct_indices` path compiles, so the counterpart can currently
   only be validated on degenerate windows (`active <= sparse_count`).

3. **UB budget** — the Ascend UB is 1 572 864 bits = 192 KiB. The rewrite's
   `gqa` and `sparse_attn_sharedkv` compressed branches exceed it; see §9 for
   how to read the failure out of `ir/` (a specialisation with `ttir` +
   `ttadapter` but **no `npubin`** failed to compile). `gqa.run` and
   `kv_rmsnorm_rope_cache.run` pass
   `multibuffer=True, limit_auto_multi_buffer_of_local_buffer='no-limit'`,
   which lets the compiler multi-buffer without a cap — the first knob to try
   when a tile is "just" over the limit.

## 12. Reading the IR without the artifacts

`ir/<op>/<impl>/<src-hash>/<kernel>.<ext>` — stage order is
`ttir → ttadapter (CommonIR→Linalg) → mlirbc → bcmlir → npubin`, so:

* `ttir` vs `ttadapter` = what the CommonIR/TLE primitives lowered into.
* **missing `npubin`** = that specialisation did not build.
* `ir_bundle/<op>__<impl>.{ttir,ttadapter}.mlir` concatenates every kernel of
  one `(op, impl)`, which is what to `diff` when comparing the rewrite against
  the counterpart.
* For the Linalg tile shapes, grep the `bcmlir`/`ttadapter` for
  `memref<...>` (UB buffers) — that is where an overflow is visible
  (e.g. `gqa` D=128 emits 17 × `memref<128x128xbf16>` = 557 KiB of UB
  buffers against a 192 KiB budget).

4. **`num_stages > 2` in the counterpart's flash kernel** — once (1) is worked
   around, `gqa`'s counterpart fails at *codegen*:
   `AssertionError('Only `range` iterator supports num_stages <= 2')` on
   `for col_start in tl.range(..., num_stages=num_stages)` in
   `flash_kernel.py`. `flash_fwd_kernel` takes `num_stages` from its tuned
   autotune configs (`runtime.get_tuned_config("attention")`) and
   `flash_fwd_splitkv_kernel` from a `lambda args: 3` heuristic.
   `_patch_flash_num_stages(2)` caps both; the run log records it under
   `meta.notes` (and hence in the summary's "Workarounds applied" section),
   because the counterpart is then measured with a shallower pipeline.
   *If that section is missing on your next run, the patch did not find the
   config objects and you are looking at the unpatched failure.*

5. **Experimenting with tile sizes without editing the kernel** —
   `--new-const _BLOCK_N=64` sets module-level constants on the **CommonIR**
   kernel before the call. This is a diagnostic, not a measurement of the
   committed code: it is for answering "is that UB overflow just a too-large
   tile?" (e.g. `gqa` `_BLOCK_M/_BLOCK_N`, `sparse_attn_sharedkv`
   `_CMP_CHUNK/_N_CHUNK/_HEAD_BLOCK`). The override is echoed in `meta.json`
   and in the summary so such rows can never be mistaken for the real kernel.

### Artifact size after `--artifacts failures`

First run: 1.91 GB of `.pt` (95 % of it `compressor`'s fp32 `state_cache`,
546 MB for a **24-token** decode case). Second run: **34 MB**, and only for the
cases that did not compare — plus a 12 MB tarball. Artifacts are now also
dropped when only one of the two impls produced one (a lone `.pt` is useless
to `compare`).

### Correction (run `fix2`): where the num_stages patch went wrong

`_patch_flash_num_stages` first looked for `obj.heuristics`, but this Triton
stores the dict as **`Heuristics.values`** (`python/triton/runtime/autotuner.py`),
so the patch silently did nothing — and because the note was only appended
after a *successful* call, the run showed no trace of it. Both are fixed:
the patcher accepts `.values` and `.heuristics`, dedupes the capped configs
(the tuned list enumerates stages 2/3/4, which collapse), and notes are now
collected in a process-global `_RUN_NOTES` that is written even when the call
raises. The `gqa` ref is also launched with `disable_splitkv=True`: the
split-kv path needs `get_device_properties(...).multi_processor_count`, which
torch_npu does not expose, and its kernel's `num_stages` comes only from the
`lambda args: 3` heuristic.

**Check `## Workarounds applied` in `summary.md` before interpreting any
`gqa` row** — its absence means no patch was applied.

### Round 3 results: what was actually solved

* **`sparse_attn_sharedkv` compressed branch — SOLVED by `HEAD_BLOCK` 32 → 16.**
  `--new-const _HEAD_BLOCK=16` makes CFA *and* SCFA compile; accuracy is 1 ULP
  of bf16 against the counterpart on all three compressed cases. The `[HEAD_BLOCK, *]`
  tensors (`acc` is `[32,512]` fp32 = 64 KiB) are live for the whole kernel,
  whereas the `[128,512]` PV tiles are streamed — so halving `HEAD_BLOCK` frees
  the ~80 KiB that was missing. The `_CMP_CHUNK` probe made it *worse*
  (208 → 280 KiB for cfa, and scfa was bit-identical before/after), which is
  what ruled the cmp tiles out.
* **`gqa` rewrite — compiles at `_BLOCK_M=64,_BLOCK_N=64`**, including both
  `D=128` cases that overflowed at 128×128.
* **`gqa` counterpart — still unbuildable.** The `num_stages` patch now
  demonstrably applies (see `## Workarounds applied`), yet
  `flash_fwd_kernel` keeps failing at the same `tl.range(..., num_stages=...)`
  line, i.e. the installed `flaggems_vllm` does not expose the config objects
  the repo copy does. `_clamp_num_stages_globally()` additionally clamps the
  `num_stages` kwarg in every `JITFunction.run`, and `_call_ref_gqa` now falls
  back to a **torch SDPA golden** on a compilation error, so `gqa` accuracy no
  longer depends on that kernel building.

  Note for the fallback: `is_causal=True` in `F.scaled_dot_product_attention`
  is **upper-left** aligned; `gqa.py` is bottom-right, so the golden uses
  `torch.nn.attention.bias.causal_lower_right(S_q, S_kv)`. `selfcheck` verifies
  the golden against a naive bottom-right loop (including `S_q != S_kv`).
  Fallback rows are marked in `## Workarounds applied` — **their `ref (ms)` is a
  vendor SDPA number, not a Triton kernel**, so do not quote it as a speedup.

---

## 13. Performance investigation (why the rewrite is slower, and how to fix it)

### The comparison was never like-for-like

Both implementations include host cost in the wall time, and the counterpart's
host cost is huge: its time is **flat at ~1.0 ms** for `causal_conv1d_fn` from
8 to 1024 tokens, and **~1.5 ms** for `compressor` from 24 to 1024 tokens. So
where the rewrite "wins" 2–3x on small shapes that is host overhead, and where
it loses by 2x the gap is on-device and real.

`--measure-mode npu-graph` captures the call once and times `graph.replay()`,
giving **device-only** milliseconds. Use it for every perf claim:

```bash
python commonir_bench.py run --outdir out/graph --cases default --measure-mode npu-graph
```

If a worker reports `graph_capture_failed` it means the op synchronises the host
inside `run()` (e.g. `causal_conv1d_fn` does `.cpu().tolist()` on
`query_start_loc`) — that is itself a finding, and the worker falls back to loop
timing.

### Structural differences found by reading the two implementations

**`compressor`** — the counterpart runs *one* persistent projection kernel for
both halves (`_project_concat_kernel`, weight `cat(wkv_t, wgate_t)`,
`NUM_PROGRAMS = 24` = one program per core), and derives
`BLOCK_K = floor_pow2(min(ceil_pow2(H), (L1 - intercept) // slope))` from an
explicit L1/L0C resource model — so K is 256–512 for `H = 4096`. The rewrite
launches `_proj_scatter_kernel` **twice** (once per weight matrix, re-reading
all of `x`) with `BM/BN/BK = 128/128/128` hardcoded. 32 K-steps of a
128x128x128 dot instead of ~8 of 128x256x512 is the classic Ascend GEMM
efficiency cliff; measured 13 TFLOP/s for the rewrite against ~52 for the
counterpart.

**`causal_conv1d_fn`** — the counterpart keeps the `width-1` history in
registers/UB and slides it, so `x` is read **once** (`N*D`). The rewrite does
`for j in range(WIDTH)` `tle.dsa.copy` per 32-token chunk, so `x` is read
**WIDTH times** (`4*N*D`), in 128x32x2 = 8 KiB DMA transfers.

### Tuning without editing kernels

`_BLOCK_T`, `_MAX_BLOCK_D` (causal_conv1d_fn) and `_PROJ_BM_MAX`, `_PROJ_BN`,
`_PROJ_BK` (compressor) are now module-level constants, so `--new-const` can
sweep them; `sweep` runs a grid, times each variant, and **rejects any variant
that changes the accuracy verdict**:

```bash
# compressor: is BK/BN the 4x?
python commonir_bench.py sweep --outdir out/sw_comp --ops compressor \
  --cases prefill0_b2_rank3,prefill1_hd128,prefill_small,decode3 \
  --sweep "_PROJ_BK=128|256|512,_PROJ_BN=128|256" --measure-mode npu-graph

# causal_conv1d_fn: bigger DMA transfers / more programs
python commonir_bench.py sweep --outdir out/sw_cc1d --ops causal_conv1d_fn \
  --cases b10_d4096_s1024_pad_fp16,b4_d4096_s249 \
  --sweep "_BLOCK_T=32|64|128,_MAX_BLOCK_D=64|128" --measure-mode npu-graph

# sparse: HEAD_BLOCK=32 fails to build, so 16 is the baseline; N_CHUNK is the
# other (PV-tile width) knob
python commonir_bench.py sweep --outdir out/sw_sas --ops sparse_attn_sharedkv \
  --cases cfa_prefill_small,scfa_decode \
  --sweep "_HEAD_BLOCK=16|32,_N_CHUNK=64|128" --measure-mode npu-graph
```

Output: `sweep.json`, `sweep.md` (full grid + *best accuracy-preserving variant
per case*). The `ref@base` run for `compressor` also sets
`COMPRESSOR_PRINT_RESOURCE_PLAN=1`, so `logs/*ref@base.log` records the
counterpart's own `BLOCK_K` plan — the number the rewrite needs to match.

### Still-untested structural fix

The two-launch split in `compressor` cannot be reached by a constant: it needs
`_proj_scatter_kernel` to take a `[2*proj_dim, hidden]` concatenated weight and
emit both halves in one launch (what `_project_concat_kernel` does). If the BK/BN
sweep does not close the 4x gap, that is the next change.

### Round 4 measurements (`out/graph`, `out/sw_comp`, `out/sw_cc1d`)

**Method bug found and fixed (twice):** most *counterparts* cannot be graph
captured (`RuntimeError: copy_between_host_and_device`) because they sync the
host inside the call, so `out/graph` mixed device-only `new` numbers against
host-inclusive `ref` numbers — e.g. `kv_rmsnorm_rope_cache/decode_bf16 344x` is
an artefact (0.066 device vs 22.7 host+device). Fixes: `sweep` now drops the
whole case to `loop` if the ref cannot be captured, `report` shows `mode!`
instead of a ratio when the two sides' modes differ, and `--measure-mode
host-split` attributes host vs device when *neither* side is capturable.

**`--measure-mode host-split`** times `run()` normally, then times it again with
every `JITFunction.run` stubbed out. The second number is the host cost
*including* the D2H sync (`causal_conv1d_fn` does `.cpu().tolist()` on
`query_start_loc` — which is also why it cannot be graph-captured). Use it for
exactly those ops:

```bash
python commonir_bench.py run --outdir out/hspl --ops causal_conv1d_fn \
  --cases b10_d4096_s1024_pad_fp16,b4_d4096_s249 --measure-mode host-split
```

**Results so far**

* `compressor` — `BN=256` is worth 27 % on the worst case (`prefill0_b2_rank3`
  5.20 → 3.79 ms), `BN=256+BK=256` 31 % (→ 3.61 ms), `BK=512` adds nothing above
  `BK=256` and `BK=512+BN=256` does not compile. The counterpart's own resource
  plan (`logs/*ref@base.log`) says it runs `block_k=512` with `l1_bytes=524288`,
  i.e. **100 % of L1**; the rewrite's 128/128/128 uses 131 KiB (25 %). 31 % is
  real but the remaining ~3x is structural: 2 launches re-reading `x` and a
  per-tile (not persistent) grid.
* `causal_conv1d_fn` — tiling is **not** the cause: `_BLOCK_T=64` is worth 8 %
  (2.13 → 1.95 ms) and reaches parity on the smaller case; `_BLOCK_T=128` is
  worse than 64 and `_MAX_BLOCK_D=64` is worse than 128, so the kernel is not
  parallelism-starved and not simply DMA-size limited. The likely cause is the
  body loop's `dsa.copy → to_tensor → acc` chain being fully serial (no
  double buffering) — `success_case/matmul_double_buffer.py` is the in-tree
  template for that fix.
* `gqa` — the `JITFunction.run` `num_stages` clamp made the real counterpart
  compile at last: `b1_s1024_h8x2_d64_noncausal` **new 0.157 ms vs ref 0.193 ms
  (1.23x), max_abs_err 1.2e-4** on 524 288 elements — the first gqa comparison.
* `~62 µs` is the graph-replay floor on this box (`sparse/swa_decode` and
  `gqa/b1_s128` both land on it), so sub-0.1 ms graph numbers are not
  comparable; use loop or host-split there.

### Round 5: host/device attribution and the tile ceiling

**`causal_conv1d_fn` — host is ruled out; the gap is device, and it is
size-dependent.** `--measure-mode host-split` (stubs every `JITFunction.run`):

| case | impl | total | host | device |
|---|---|---|---|---|
| `b10_d4096_s1024_pad_fp16` | new | 2.096 | 0.242 | **1.854** |
| `b10_d4096_s1024_pad_fp16` | ref | 1.018 | 0.383 | **0.635** |
| `b4_d4096_s249` | new | 1.084 | 0.241 | **0.843** |
| `b4_d4096_s249` | ref | 1.003 | 0.387 | **0.616** |

The rewrite's host cost is *lower* than the counterpart's (0.24 vs 0.38 ms), so
the two `.cpu().tolist()` calls are not the problem. The ref's device time is
essentially **flat** (0.616 → 0.635 for 4.1x the tokens — its `BLOCK_M=128`
token loop is a serial dependent chain) while the rewrite **scales**
(0.843 → 1.854). So the rewrite loses at small token counts and would overtake
the counterpart at larger ones; `causal_conv1d_fn/b10_d4096_s4096_pad_fp16` (in
the `full` set) is the case that decides it.

The residual device cost is most likely the **head path**: `head_dsa` is enabled
exactly when `total_len / nseq < 256` (78 here), and it emits, per program,
`width-1` unrolled output columns × `WIDTH` taps of `(BLOCK_D,)`-sized masked
`tl.load`s plus `width-1` separate column stores — ~12 tiny ops per program on a
416-program grid. That also explains why `_BLOCK_T` only bought 8%: it shrinks
the *body* only. `_HEAD_DSA_AVG_LIMIT` is now a module constant, so this is
directly testable (`0` forces the masked single-tile head).

**`compressor` — `BN=256, BK=256` is accuracy-verified and the best variant.**

| case | committed 128/128/128 | BN=256 BK=256 | ref | verdict |
|---|---|---|---|---|
| `prefill0_b2_rank3` | 5.234 | **3.570** (−32%) | 2.711 | 0.54x → **0.76x** |
| `prefill1_hd128` | 2.045 | **1.622** (−21%) | 1.709 | 0.83x → **1.05x (now faster)** |
| `prefill_small` | 0.744 | 0.809 | 1.471 | 1.82x |

`max_abs_err = 0` on all three, so this change is safe to adopt.

`BN=512` is **impossible**, and the reason is the L0C accumulator, not L1:
`cc overflow, requires 2097152 bits while 1048576 bits available` — L0C is
128 KiB, so a `[BM,BN]` fp32 accumulator caps at `BM*BN <= 32768` elements
(128x256 exactly fills it). The width-in-one-tile variant therefore requires
`BM=64, BN=512` (L0C exactly 128 KiB, L1 288 KiB at BK=128) — which is also the
cleanest test of whether the remaining 1.3x on `prefill0_b2_rank3` is the
per-tile grid (1 N-tile => no A-tile re-read, 4x fewer programs) or the fused
data-dependent scatter.

### Round 6: root causes confirmed, two fixes applied

**`causal_conv1d_fn` — the head path is NOT the cause; global traffic is.**

`_HEAD_DSA_AVG_LIMIT=0` (masked single-tile head) is *worse*: 2.450 vs 1.926 ms
total at 1024 tokens, so the per-column DSA head pays for itself. With
`_BLOCK_T=64` the device split is:

| tokens | new device | ref device | ratio | new traffic | ref traffic | new GB/s | ref GB/s |
|---|---|---|---|---|---|---|---|
| 249 | 0.771 | 0.629 | 1.23x | 10.2 MB | 4.1 MB | 13 | 6.5 |
| 1024 | 1.671 | 0.643 | 2.60x | 41.9 MB | 16.8 MB | 25 | 26 |
| 4096 | 2.491 | 0.722 | 3.45x | 167.8 MB | 67.1 MB | 67 | 93 |

Reads `x` `WIDTH` times, so it moves **2.5x the bytes** at comparable achieved
bandwidth — that is the whole regression. (The ref's device time is nearly flat
because its `BLOCK_M=128` token loop is a serial dependent chain, so it does not
scale either; the rewrite loses on traffic but would keep scaling.)

*Fix applied:* `_BODY_HALO = 1` cuts every tap of a chunk from a **single loaded
strip**, halving `x` traffic (2·BLOCK_T tokens loaded per BLOCK_T outputs —
`tl.arange` forces a power of two, so 2x is the reachable factor without a
masked store path). It is a separate statically-known loop followed by the
unchanged per-tap loop for the chunks whose strip would cross the part
boundary, because a runtime branch around the accumulator is the pattern this
backend miscompiles. **Default stays 0 (unchanged behaviour)**; verified locally
for bounds, coverage and *bit-identical* accumulation order, but not yet on the
NPU — enable with `--new-const _BODY_HALO=1`.

**`compressor` — `BN` is the lever, `BK` is not; the default is now shape-gated.**

`BM=64/BN=512/BK=128` (L0C exactly 128 KiB) is the best on the worst case
(3.473 ms) but only ~5 % ahead of `BM=128/BN=256/BK=256` (3.633) and it regresses
`prefill_small`. Adopted instead, mirroring the counterpart's own
`block_n = 256 if wide_projection else 128`:

```python
_PROJ_BN = 256            # upper bound
_PROJ_BK = 256
_PROJ_WIDE_MIN_TOKENS = 2048   # below this, BN=128 measured better
```

All of these are `max_abs_err = 0` against the counterpart (out/sw_comp2,
out/sw_comp3), i.e. bit-exact. Net: `prefill0_b2_rank3` 5.234 → 3.633 ms
(0.54x → 0.77x), `prefill1_hd128` 2.045 → 1.670 ms (0.83x → 1.03x),
`prefill_small` and the decode cases unchanged.

The residual ~1.3x on `prefill0_b2_rank3` is structural and NOT reachable by
tiling (the counterpart at the same 128/128/512 tiles is still 3.6x faster): the
rewrite fuses a data-dependent scatter store into the matmul
(`dst = cache_ptr + row[:, None] * W2 + cols`) where the counterpart writes a
dense `state_proj` workspace and scatters separately, and it runs 256 one-tile
programs against the counterpart's 24 persistent ones. Splitting the scatter out
is the next change, and it needs its own validation round.

### Round 7: the `_BODY_HALO` bugs (both mine) and the compressor confirmation

**`compressor`: the new default is confirmed end-to-end** (`out/after_comp`,
`--measure-mode npu-graph`, all five cases `max_abs_err = 0`):

| case | before (graph) | after (graph) |
|---|---|---|
| `prefill0_b2_rank3` | 5.179 | **3.547 (−31.5%)** |
| `prefill1_hd128` | 2.045 | **1.666 (−18.5%)** |
| `prefill_small` | 0.744 | 0.722 |
| `decode3` | 0.133 | 0.123 |
| `decode5_c128` | 0.088 | 0.082 |

That run also shows `mode!` in every ratio, because the counterpart cannot be
graph-captured — so `produce` now applies the same reconciliation `sweep` does:
if one side of a case was captured and the other was not, the capturable side is
re-measured in `loop` mode. Plain `run --measure-mode npu-graph` therefore
produces same-kind numbers for every op, instead of ratios that mix device-only
with host+device.

**`_BODY_HALO` failed on its first attempt, in two ways, both my bugs:**

1. `ub overflow, requires 1703936 bits while 1572864 available` — I converted the
   whole `[BLOCK_D, 2*BLOCK_T]` strip to fp32 (64 KiB) instead of extracting the
   `[BLOCK_D, BLOCK_T]` window first. Fixed: extract in the native dtype, then
   widen. Also means the halo path wants `_BLOCK_T=32`, not 64.
2. `aivec error / mte error` — a **runtime** vector-core fault on the 4096-token
   case (it compiled, then read out of range): `nhalo =
   (be - bs - (2*BLOCK_T - H)) // BLOCK_T + 1` uses Triton's `//`, which
   **truncates toward zero** unlike Python's floor, so a negative numerator gave
   `nhalo = 1` and a strip far past the end of the part. Fixed with an explicit
   `tl.where(room >= 0, ...)`. Emulating C-style division over 1–8 `BLOCK_T`
   spans, the old formula was wrong in 46 of the sampled cases (strips up to 25
   tokens past the part); the new one never is.

Both are fixed and re-verified locally (bounds, coverage, bit-identical
accumulation order, and the `nhalo` arithmetic under truncating division), but
**`_BODY_HALO` still defaults to 0** — it needs one clean NPU run before it can
be the default. Run it with `_BLOCK_T=32`:

```bash
python commonir_bench.py sweep --outdir out/halo2 --ops causal_conv1d_fn \
  --cases b4_d4096_s249,b10_d4096_s1024_pad_fp16,b10_d4096_s4096_pad_fp16 \
  --sweep "_BODY_HALO=1,_BLOCK_T=32" --measure-mode host-split
```

### Round 8: `_BODY_HALO` faults the MTE engine — diagnosis and a second form

`_BODY_HALO=1,_BLOCK_T=32` now compiles and then faults on **all three** cases
at runtime:

```
aivec error ... mte error info: 0x11060000bd   (identical code on every case)
```

So it was not the `nhalo` arithmetic (that bug was real and is fixed, but this
fault is independent). The consistent MTE error points at
`tle.dsa.extract_slice(strip, [0, j], [BLOCK_D, BLOCK_T], [1, 1])`: a **strided
column window out of a UB tile**. Every existing use of `extract_slice` in these
kernels extracts a *contiguous* row (`[j, 0], [1, BLOCK_D]`), and the only other
strided use in the tree (`extract_slice(x, (0,0), (B,HALF), (1,2))` in the
counterpart's RoPE kernel) strides a *register* tensor, not UB.

Added `_BODY_HALO=2`, which performs the same shift with
`tl.gather(strip, kt[None, :] + j, axis=1)` — a register permute, the same
primitive the compressor kernel already uses for its RoPE partner. Both forms
can be tested in one sweep because a faulting variant only kills its own worker
(the harness isolates it; subsequent cases still ran clean in round 7).

```bash
python commonir_bench.py sweep --outdir out/halo3 --ops causal_conv1d_fn \
  --cases b4_d4096_s249,b10_d4096_s1024_pad_fp16,b10_d4096_s4096_pad_fp16 \
  --sweep "_BODY_HALO=1|2,_BLOCK_T=32" --measure-mode host-split
```

**If `_BODY_HALO=2` also fails**, the conclusion is that this backend has no
cheap way to shift within a UB-resident tile, and the honest options for
`causal_conv1d_fn` are (a) accept 0.39–0.5x on large token counts, noting that
the counterpart's half of that gap is a serial-latency floor that does not
scale, or (b) restructure the body around an explicit UB-resident sliding window
(mirroring what the counterpart does per token, but 32 tokens at a time) — a
much larger change than a tiling constant.
