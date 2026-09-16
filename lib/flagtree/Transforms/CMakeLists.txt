# FlagTree-owned TTIR passes. Kept out of lib/Dialect/Triton/Transforms so
# upstream rebases do not conflict. The passes are declared in
# include/flagtree/Transforms/Passes.td (tablegen target
# FlagTreeTransformsIncGen); see include/flagtree/Transforms/Passes.h.
add_triton_library(FlagTreeTransforms
  LaneVectorize.cpp

  DEPENDS
  FlagTreeTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRTensorDialect
  TritonIR
)
