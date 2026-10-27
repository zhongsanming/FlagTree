// fold_reduce_rows: one repeat-parameterised reduce that produces one result
// per row, after folding each row's two halves together.
//
// Why this op exists
// ------------------
// Measured on this backend (KDA_ASCENDC_VS_TLE.md 6.5-6.10), a cross-lane
// reduction costs ~480-650 cycles regardless of how many lanes it covers, and
// the KDA prefill kernel needs four of them per token: two for the q/k L2 norms
// and two for the state (v and out). Those four are 66 % of its wall clock.
//
// Two of them can be merged into one whenever both results come from the same
// shape of row, which is exactly the case here:
//
//   * s_q and s_k   -> both are 128-lane sums of q*q and k*k
//   * v_corr and out_part -> both are 128-lane sums over the same state tile
//     (the second needs the identity out = sum(state_d*q) + (k.q)*v, which is
//     what made an earlier attempt at this slower: without a fused reduce the
//     identity needs a fifth reduction. See 6.6.)
//
// WholeReduceSum already takes a repeat count, so one instruction can produce
// many results; what was missing was a way to reach it from Triton, where a
// tl.sum over C rows is C independent reductions and there is no way to say
// "these rows are one instruction". tl.join/tl.split cannot express it either
// (that path does not compile here).
//
// Semantics
// ---------
//   src : [repeat, 2*mask] in UB, row-major, row stride src_rep_stride blocks
//   dst : [repeat]        in UB, contiguous (one result per row)
//   tmp : [repeat, mask]  in UB scratch, supplied by the caller
//
//   dst[r] = sum_{i < mask} ( src[r, i] + src[r, i + mask] )
//
// Two instructions: a strided vadd that folds the halves into tmp, then one
// vcadd over all `repeat` rows. MODE=0 so each repeat writes its own result
// instead of accumulating into SPR.
//
// fp32 and fp16 are registered; bf16 has no vcadd on this core (bfloat16_t is
// accepted only for the fold, so it is left out rather than silently wrong).

#include "Utils.h"

__aiv__ static inline __attribute__((always_inline)) void
set_mask(int32_t len, int32_t dtype_size) {
  const int32_t DEFAULT_BLOCK_SIZE = 256;
  const uint64_t FULL_MASK = 0xffffffffffffffff;
  int32_t type_len = DEFAULT_BLOCK_SIZE / dtype_size;
  const int32_t half_type_len = 64; // 1 register -> 64 bits -> 64 elements
  const int32_t len_coeff = 2;      // 2 registers for masks
  if (len == half_type_len) {
    set_vector_mask(0, FULL_MASK);
    return;
  } else if (len == type_len || len >= half_type_len * len_coeff) {
    set_vector_mask(FULL_MASK, FULL_MASK);
    return;
  }
  set_vector_mask(static_cast<uint64_t>(
                      (len > half_type_len)
                          ? (((static_cast<uint64_t>(1))
                              << static_cast<uint32_t>(len - half_type_len)) -
                             1)
                          : 0),
                  static_cast<uint64_t>((len > half_type_len)
                                            ? FULL_MASK
                                            : (((static_cast<uint64_t>(1))
                                                << static_cast<uint32_t>(len)) -
                                               1)));
}

template <typename T>
__aiv__ __attribute__((always_inline)) void
fold_reduce_rows_impl(__ubuf__ T *dst, __ubuf__ T *tmp, __ubuf__ T *src,
                      const int32_t repeat_times, const int32_t mask,
                      const int32_t src_rep_stride) {
  // tmp is written tightly: one row of `mask` lanes, so the next row starts
  // mask * sizeof(T) / 32 datablocks later.
  const int32_t tmp_rep_stride =
      static_cast<int32_t>(mask * static_cast<int32_t>(sizeof(T)) / 32);

  // Fold: tmp[r][i] = src[r][i] + src[r][i + mask].  Block strides are 1 (the
  // lanes of a row are consecutive datablocks); the two sources advance by the
  // same row stride, one of them starting `mask` lanes in.
  set_mask(mask, sizeof(T));
  vadd(tmp, src, src + mask, repeat_times,
       /*dst_blk*/ 1, /*src0_blk*/ 1, /*src1_blk*/ 1,
       /*dst_rep*/ tmp_rep_stride, /*src0_rep*/ src_rep_stride,
       /*src1_rep*/ src_rep_stride);
  pipe_barrier(PIPE_V);

  // One reduce for all rows.  MODE=0: one result per repeat, into dst.
  set_mask(mask, sizeof(T));
  vcadd(dst, tmp, repeat_times,
        /*dst_rep*/ 1, /*src_blk*/ 1, /*src_rep*/ tmp_rep_stride,
        /*MODE*/ 0);
}

#define REGISTER_CIFACE_FOLD_REDUCE_ROWS(TYPE_NAME, T)                         \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_custom_fold_reduce_rows_##TYPE_NAME(                        \
          memref_t<__ubuf__ T, 1> *src, const int32_t repeat_times,            \
          const int32_t mask, const int32_t src_rep_stride,                    \
          memref_t<__ubuf__ T, 1> *dst, memref_t<__ubuf__ T, 1> *tmp) {        \
    auto src_ptr = src->aligned + src->offset;                                 \
    auto dst_ptr = dst->aligned + dst->offset;                                 \
    auto tmp_ptr = tmp->aligned + tmp->offset;                                 \
    fold_reduce_rows_impl<T>(dst_ptr, tmp_ptr, src_ptr, repeat_times, mask,    \
                             src_rep_stride);                                  \
  }

extern "C" {
REGISTER_CIFACE_FOLD_REDUCE_ROWS(half, half)
REGISTER_CIFACE_FOLD_REDUCE_ROWS(float, float)
}
