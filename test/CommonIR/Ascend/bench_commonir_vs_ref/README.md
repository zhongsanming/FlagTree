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
