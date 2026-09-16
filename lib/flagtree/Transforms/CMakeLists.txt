# flagtree-owned TTIR passes. Kept out of lib/Dialect/Triton/Transforms so
# upstream rebases do not conflict. The pass still uses the upstream tablegen
# declarations (include/triton/Dialect/Triton/Transforms/Passes.td).
add_triton_library(FlagTreeTransforms
  LaneVectorize.cpp

  DEPENDS
  TritonTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRTensorDialect
  TritonIR
)
