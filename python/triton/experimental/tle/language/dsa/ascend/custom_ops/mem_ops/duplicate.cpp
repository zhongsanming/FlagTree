#include "Utils.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void duplicate_bitwise_mask_impl(
    __ubuf__ T *dst, T scalar, __ubuf__ uint64_t *mask, const uint8_t repeat_times,
    const uint16_t dst_block_stride, const uint8_t dst_repeat_stride) {
  set_vector_mask(static_cast<uint64_t>(mask[1]),
                  static_cast<uint64_t>(mask[0]));
  vector_dup(dst, scalar, repeat_times, dst_block_stride, 1, dst_repeat_stride,
             0);
}

#define REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(TYPE_NAME, T)                   \
  __aiv__ __attribute__((always_inline)) void                                  \
      _mlir_ciface_custom_duplicate_bitwise_mask_##TYPE_NAME(                  \
          memref_t<__ubuf__ T, 1> *scalar_value,                               \
          memref_t<__ubuf__ uint64_t, 1> *mask, const uint8_t repeat_times,    \
          const uint16_t dst_block_stride, const uint8_t dst_repeat_stride,    \
          memref_t<__ubuf__ T, 1> *dst) {                                      \
    auto dst_ptr = dst->aligned + dst->offset;                                 \
    auto scalar_ptr = scalar_value->aligned + scalar_value->offset;            \
    auto mask_ptr = mask->aligned + mask->offset;                              \
    duplicate_bitwise_mask_impl<T>(dst_ptr, static_cast<T>(scalar_ptr[0]),     \
                                   mask_ptr, repeat_times, dst_block_stride,   \
                                   dst_repeat_stride);                         \
  }

extern "C" {
REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(half, half)
REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(bf16, bfloat16_t)
REGISTER_CIFACE_DUPLICATE_BITWISE_MASK(float, float)
}
