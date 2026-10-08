#include "kernel.h"

#if __has_include("sdkconfig.h")
#include "sdkconfig.h"
#endif

#if defined(__XTENSA__) && (defined(CONFIG_IDF_TARGET_ESP32S3) || defined(RUVI_KERNEL_ESP32_S3))
#define RUVI_HAS_S3 1
extern "C" int32_t ruvi_kernel_s3_dot_raw(const int8_t*, const int8_t*, size_t);
#else
#define RUVI_HAS_S3 0
#endif

namespace ruvi {
namespace kernels {
bool esp32_s3_available() { return RUVI_HAS_S3 != 0; }

Status dot_s3_aligned(const int8_t* a, const int8_t* b, size_t length, int32_t* result) {
    if (!a || !b || !result || !length || length > 32768 || (length & 15u) ||
        (reinterpret_cast<uintptr_t>(a) & 15u) || (reinterpret_cast<uintptr_t>(b) & 15u))
        return Status::invalid_argument;
#if RUVI_HAS_S3
    *result = ruvi_kernel_s3_dot_raw(a, b, length);
    return Status::ok;
#else
    return Status::unsupported;
#endif
}
} // namespace kernels
} // namespace ruvi
