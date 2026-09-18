# FlagTree-owned TTIR pass declarations. Tablegenned separately from
# include/triton/Dialect/Triton/Transforms so upstream rebases of that file do
# not conflict. `-name FlagTree` names the generated registration function
# (registerFlagTreePasses); the per-pass names come from the def names.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name FlagTree)
add_public_tablegen_target(FlagTreeTransformsIncGen)
