#ifndef FLAGTREE_TRANSFORMS_PASSES_H_
#define FLAGTREE_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

// Declarations of the FlagTree-owned TTIR passes: for each pass this expands
// to its create<Name>() factory (used by python/src/passes.cc) and, on the
// second inclusion, to registerFlagTreePasses() (used by the *-opt tools).
// See include/flagtree/Transforms/Passes.td.
#define GEN_PASS_DECL
#include "flagtree/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "flagtree/Transforms/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
