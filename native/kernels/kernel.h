#ifndef RUVI_KERNELS_KERNEL_H
#define RUVI_KERNELS_KERNEL_H

#include <cstddef>
#include <cstdint>

namespace ruvi {
namespace kernels {

enum class Status { ok, invalid_argument, insufficient_scratch, invalid_weight, unsupported };
enum class Backend { portable, automatic, esp32_s3, rv1103_neon };

// RUVI0001 weight payload (not the file header): two low-to-high bits per
// element, continuous index n*K+k. 00=-1, 01=0, 10=1; 11 is invalid.
struct Problem {
    const uint8_t* weights = nullptr;
    size_t weight_bytes = 0;
    size_t m = 0, n = 0, k = 0;
    float weight_scale = 1.0f;
    const int8_t* input = nullptr;
    size_t input_bytes = 0;
    size_t input_stride = 0; // bytes; each token has align16(k) readable bytes
    const float* multipliers = nullptr;
    size_t multiplier_count = 0;
    size_t multiplier_stride = 1; // float elements, strictly positive
    float* output = nullptr;
    size_t output_count = 0;
    size_t output_stride = 0; // float elements, at least n
};

struct Options {
    size_t tile_rows = 16; // exactly 8, 16, or 32
    size_t n_begin = 0;
    size_t n_count = SIZE_MAX; // SIZE_MAX means all remaining output rows
    Backend backend = Backend::automatic;
};

struct Scratch {
    int8_t* weights = nullptr; // private to this caller; align16 for S3
    size_t weight_bytes = 0;
    int8_t* input_copy = nullptr; // optional align16 single-token microcopy
    size_t input_copy_bytes = 0;
};

// Returns zero for invalid k/tile_rows. k must be in [1,32768].
size_t padded_k(size_t k);
size_t weight_scratch_bytes(size_t k, size_t tile_rows);
bool esp32_s3_available();
bool rv1103_neon_available();

// No allocations. Distinct N partitions may execute concurrently with distinct
// scratch and disjoint output columns. Input/weights/multipliers are read-only.
// Buffers must not overlap writable output or scratch. Multipliers must be
// finite and >0; scale finite. Padding input values are ignored (weights zero).
// Validation failures occur before any output writes. An optional input_copy
// allows S3/RV1103 with unaligned input bases/strides; otherwise align16 is required.
Status run(const Problem& problem, const Options& options, const Scratch& scratch);

// Checked EE signed-int8 dot primitive. Both buffers must be aligned to 16 and
// length a multiple of 16 in [16,32768]. No hardware access on invalid arguments.
Status dot_s3_aligned(const int8_t* a, const int8_t* b, size_t length, int32_t* result);

// Checked ARMv7-A NEON signed-int8 dot primitive for Rockchip RV1103 (Cortex-A7).
// Both buffers must be aligned to 16 and length a multiple of 16 in [16,32768].
Status dot_rv1103_aligned(const int8_t* a, const int8_t* b, size_t length, int32_t* result);

} // namespace kernels
} // namespace ruvi
#endif
