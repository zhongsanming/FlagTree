# Ascend backend on FlagTree 3.6

This branch ports the upstream triton-ascend 3.6 backend onto FlagTree's
`flag/main` (Triton 3.6) line and carries over the FlagTree 3.5 CommonIR /
TensorView / FLIR work.

## Source pins

- triton-ascend: `v3.6.0.post2` (`7ae01b08c`, `release/3.6.0`)
- AscendNPU-IR: `bb13ef6120526aa3df6f950006af5ec0e585d2f6`
- LLVM: `f6ded0be` + `third_party/ascend/patch/llvm_patch_f6ded0b.patch`
  (artifact `llvm-f6ded0be-3f5016d1-<system>`)
- FLIR: `482b60b881219cab7df987f005b9d8bf5c97cbe5`

## Layout

- `third_party/ascend/` — vendored upstream 3.6 backend.
- `third_party/ascend/spec/triton/` — Python overrides loaded through
  `python/triton/spec/ascend` (symlink to `third_party/ascend/spec`).
- `third_party/ascend/spec_cpp/` — C++/TableGen overrides:
  `TritonAttrDefs.td`, `TritonOps.td`, `Traits.cpp`.
- `third_party/ascend/python/src/ir.cc` — HF32 + `create_auto_overflow_assert`.
- `third_party/ascend/python/src/buffer_ir.cc` — `triton._C.libtriton.buffer_ir`.

## Build on NixOS

```bash
nix develop
export MAX_JOBS=4
python -m pip install -e . --no-build-isolation -v
```

The flake provides the triton-ascend-patched LLVM f6ded0be and a `ccec` shim
for building without CANN. Kernel execution still needs a real CANN /
`torch_npu` environment.

Smoke test:

```bash
python - <<'PY'
import triton
from triton.backends import backends
from triton._C.libtriton import ascend, buffer_ir
from triton._C.libtriton.ascend.passes.ttir import (
    add_commonir_to_hivm, add_tensor_view_lowering, add_commonir_tv_ptr_legalize,
)
import triton.language as tl
import triton.experimental.tle as tle
print(triton.__version__, list(backends), tl.tensor_view, tle.make_partition_view)
PY
```

`triton-mlir-opt` also registers the CommonIR passes:

```bash
build/.../third_party/ascend/bin/triton-mlir-opt \
  --commonir-to-hivm --tensor-view-lowering empty.mlir
```

## Known follow-ups

- The legacy `tle.dsa.tile_load/tile_store` C++ dialect is staged under
  `third_party/ascend/tle_dsa` but is not built yet; the TensorView path uses
  `tl.load`/`tl.store` with `tl.tensor_view`.
- Core `triton-opt` (used by the full Ascend lit suite) is not wired up yet;
  `triton-mlir-opt` is available and registers CommonIR/TensorView passes.
- The NixOS host has no NPU/torch_npu, so hardware unit tests were not run
  here. The upstream 3.6 unittest tree is preserved and the 3.5 TensorView
  tests were copied in.
