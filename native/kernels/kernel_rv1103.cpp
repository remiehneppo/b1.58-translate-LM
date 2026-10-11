#include "kernel.h"

#if (defined(__arm__) || defined(__thumb__)) && (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(RUVI_KERNEL_RV1103_NEON))
#define RUVI_HAS_RV1103_NEON 1
extern "C" int32_t ruvi_kernel_neon_dot_raw(const int8_t*, const int8_t*, size_t);
#else
#define RUVI_HAS_RV1103_NEON 0
#endif

namespace ruvi {
namespace kernels {

bool rv1103_neon_available() {
    return RUVI_HAS_RV1103_NEON != 0;
}

Status dot_rv1103_aligned(const int8_t* a, const int8_t* b, size_t length, int32_t* result) {
    if (!a || !b || !result || !length || length > 32768 || (length & 15u) ||
        (reinterpret_cast<uintptr_t>(a) & 15u) || (reinterpret_cast<uintptr_t>(b) & 15u))
        return Status::invalid_argument;
#if RUVI_HAS_RV1103_NEON
    *result = ruvi_kernel_neon_dot_raw(a, b, length);
    return Status::ok;
#else
    return Status::unsupported;
#endif
}

} // namespace kernels
} // namespace ruvi
