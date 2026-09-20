#ifndef FLAGTREE_COMMON_ENVVARS_H
#define FLAGTREE_COMMON_ENVVARS_H

#include <set>
#include <string>

namespace mlir::triton::flagtree {

// Environment variables owned by FlagTree rather than by upstream Triton.
//
// They are declared here instead of in include/triton/Tools/Sys/GetEnv.hpp so
// that upstream rebases of that file do not conflict. python/src/ir.cc merges
// this set into the upstream cache-invalidating set, so changing any of them
// still invalidates the JIT cache, exactly as for an upstream variable.
//
// Keep the Python-side readers in sync (grep for the variable name); the
// backends read these directly with os.getenv.
inline const std::string kDisableLaneVectorize =
    "TRITON_DISABLE_LANE_VECTORIZE";

<<<<<<< Updated upstream
// Opt-in: enable block mode (SLP over straight-line code) in the
// triton-lane-vectorize pass. Loop mode is always on; block mode is disabled
// by default. See lib/flagtree/Transforms/LaneVectorize.cpp.
inline const std::string kEnableLaneVectorizeBlockMode =
    "TRITON_ENABLE_LANE_VECTORIZE_BLOCK_MODE";
=======
// Opt-in relaxations for the correctness guards in triton-lane-vectorize. Both
// default OFF (the guarded/safe behavior); set to 1/true/on to restore the
// aggressive transform. See lib/flagtree/Transforms/LaneVectorize.cpp.
//
//   ALLOW_CONCAT         permit the generic tensor.concat packing fallback,
//                        which materializes its operands (an allocation the
//                        bundled Ascend BiSheng/HIVM pipeline cannot lower).
//   ALLOW_ADDRESS_CONES  permit packing integer/index cones that feed memory
//                        addresses (unsafe for strided/non-contiguous access).
inline const std::string kAllowLaneVectorizeConcat =
    "TRITON_LANE_VECTORIZE_ALLOW_CONCAT";
inline const std::string kAllowLaneVectorizeAddressCones =
    "TRITON_LANE_VECTORIZE_ALLOW_ADDRESS_CONES";
>>>>>>> Stashed changes

inline const std::set<std::string> CACHE_INVALIDATING_ENV_VARS = {
    // clang-format off
    kDisableLaneVectorize,
<<<<<<< Updated upstream
    kEnableLaneVectorizeBlockMode,
=======
    kAllowLaneVectorizeConcat,
    kAllowLaneVectorizeAddressCones,
>>>>>>> Stashed changes
    // clang-format on
};

} // namespace mlir::triton::flagtree

#endif
