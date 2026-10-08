#include "kernel.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace ruvi {
namespace kernels {
namespace {
bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }
bool span_fits(size_t rows, size_t stride, size_t width, size_t capacity) {
    return width <= capacity && (rows - 1) <= (capacity - width) / stride;
}
uint8_t code_at(const uint8_t* weights, size_t index) {
    return static_cast<uint8_t>((weights[index / 4] >> (2 * (index % 4))) & 3u);
}
int32_t dot_portable(const int8_t* a, const int8_t* b, size_t k) {
    int32_t sum = 0;
    for (size_t i = 0; i < k; ++i) sum += int32_t(a[i]) * int32_t(b[i]);
    return sum;
}

// 256-entry table mapping 1 packed byte (4 2-bit codes) to 4 signed int8_t values.
// Code mapping: 00 -> -1 (0xFF), 01 -> 0 (0x00), 10 -> +1 (0x01), 11 -> invalid.
static const uint32_t s_unpack_lut[256] = {
    0xffffffffu, 0xffffff00u, 0xffffff01u, 0xffffff02u, 0xffff00ffu, 0xffff0000u, 0xffff0001u, 0xffff0002u,
    0xffff01ffu, 0xffff0100u, 0xffff0101u, 0xffff0102u, 0xffff02ffu, 0xffff0200u, 0xffff0201u, 0xffff0202u,
    0xff00ffffu, 0xff00ff00u, 0xff00ff01u, 0xff00ff02u, 0xff0000ffu, 0xff000000u, 0xff000001u, 0xff000002u,
    0xff0001ffu, 0xff000100u, 0xff000101u, 0xff000102u, 0xff0002ffu, 0xff000200u, 0xff000201u, 0xff000202u,
    0xff01ffffu, 0xff01ff00u, 0xff01ff01u, 0xff01ff02u, 0xff0100ffu, 0xff010000u, 0xff010001u, 0xff010002u,
    0xff0101ffu, 0xff010100u, 0xff010101u, 0xff010100u, 0xff0102ffu, 0xff010200u, 0xff010201u, 0xff010202u,
    0xff02ffffu, 0xff02ff00u, 0xff02ff01u, 0xff02ff02u, 0xff0200ffu, 0xff020000u, 0xff020001u, 0xff020002u,
    0xff0201ffu, 0xff020100u, 0xff020101u, 0xff020102u, 0xff0202ffu, 0xff020200u, 0xff020201u, 0xff020202u,
    0x00ffffffu, 0x00ffff00u, 0x00ffff01u, 0x00ffff02u, 0x00ff00ffu, 0x00ff0000u, 0x00ff0001u, 0x00ff0002u,
    0x00ff01ffu, 0x00ff0100u, 0x00ff0101u, 0x00ff0102u, 0x00ff02ffu, 0x00ff0200u, 0x00ff0201u, 0x00ff0202u,
    0x0000ffffu, 0x0000ff00u, 0x0000ff01u, 0x0000ff02u, 0x000000ffu, 0x00000000u, 0x00000001u, 0x00000002u,
    0x000001ffu, 0x00000100u, 0x00000101u, 0x00000102u, 0x000002ffu, 0x00000200u, 0x00000201u, 0x00000202u,
    0x0001ffffu, 0x0001ff00u, 0x0001ff01u, 0x0001ff02u, 0x000100ffu, 0x00010000u, 0x00010001u, 0x00010002u,
    0x000101ffu, 0x00010100u, 0x00010101u, 0x00010102u, 0x000102ffu, 0x00010200u, 0x00010201u, 0x00010202u,
    0x0002ffffu, 0x0002ff00u, 0x0002ff01u, 0x0002ff02u, 0x000200ffu, 0x00020000u, 0x00020001u, 0x00020002u,
    0x000201ffu, 0x00020100u, 0x00020101u, 0x00020102u, 0x000202ffu, 0x00020200u, 0x00020201u, 0x00020202u,
    0x01ffffffu, 0x01ffff00u, 0x01ffff01u, 0x01ffff02u, 0x01ff00ffu, 0x01ff0000u, 0x01ff0001u, 0x01ff0002u,
    0x01ff01ffu, 0x01ff0100u, 0x01ff0101u, 0x01ff0102u, 0x01ff02ffu, 0x01ff0200u, 0x01ff0201u, 0x01ff0202u,
    0x0100ffffu, 0x0100ff00u, 0x0100ff01u, 0x0100ff02u, 0x010000ffu, 0x01000000u, 0x01000001u, 0x01000002u,
    0x010001ffu, 0x01000100u, 0x01000101u, 0x01000102u, 0x010002ffu, 0x01000200u, 0x01000201u, 0x01000202u,
    0x0101ffffu, 0x0101ff00u, 0x0101ff01u, 0x0101ff02u, 0x010100ffu, 0x01010000u, 0x01010001u, 0x01010002u,
    0x010101ffu, 0x01010100u, 0x01010101u, 0x01010102u, 0x010102ffu, 0x01010200u, 0x01010201u, 0x01010202u,
    0x0102ffffu, 0x0102ff00u, 0x0102ff01u, 0x0102ff02u, 0x010200ffu, 0x01020000u, 0x01020001u, 0x01020002u,
    0x010201ffu, 0x01020100u, 0x01020101u, 0x01020102u, 0x010202ffu, 0x01020200u, 0x01020201u, 0x01020202u,
    0x02ffffffu, 0x02ffff00u, 0x02ffff01u, 0x02ffff02u, 0x02ff00ffu, 0x02ff0000u, 0x02ff0001u, 0x02ff0002u,
    0x02ff01ffu, 0x02ff0100u, 0x02ff0101u, 0x02ff0102u, 0x02ff02ffu, 0x02ff0200u, 0x02ff0201u, 0x02ff0202u,
    0x0200ffffu, 0x0200ff00u, 0x0200ff01u, 0x0200ff02u, 0x020000ffu, 0x02000000u, 0x02000001u, 0x02000002u,
    0x020001ffu, 0x02000100u, 0x02000101u, 0x02000102u, 0x020002ffu, 0x02000200u, 0x02000201u, 0x02000202u,
    0x0201ffffu, 0x0201ff00u, 0x0201ff01u, 0x0201ff02u, 0x020100ffu, 0x02010000u, 0x02010001u, 0x02010002u,
    0x020101ffu, 0x02010100u, 0x02010101u, 0x02010102u, 0x020102ffu, 0x02010200u, 0x02010201u, 0x02010202u,
    0x0202ffffu, 0x0202ff00u, 0x0202ff01u, 0x0202ff02u, 0x020200ffu, 0x02020000u, 0x02020001u, 0x02020002u,
    0x020201ffu, 0x02020100u, 0x02020101u, 0x02020102u, 0x020202ffu, 0x02020200u, 0x02020201u, 0x02020202u,
};
} // namespace

size_t padded_k(size_t k) { return k && k <= 32768 ? (k + 15) & ~size_t(15) : 0; }
size_t weight_scratch_bytes(size_t k, size_t rows) {
    return rows == 8 || rows == 16 || rows == 32 ? rows * padded_k(k) : 0;
}

Status run(const Problem& p, const Options& o, const Scratch& s) {
    const size_t pk = padded_k(p.k);
    const size_t needed = weight_scratch_bytes(p.k, o.tile_rows);
    const size_t max = std::numeric_limits<size_t>::max();
    if (!pk || !needed || !p.m || !p.n || !p.weights || !p.input ||
        !p.output || !p.multipliers || !s.weights || !std::isfinite(p.weight_scale) ||
        p.n > max / p.k || o.n_begin > p.n || p.input_stride < pk ||
        p.output_stride < p.n || !p.multiplier_stride ||
        (o.backend != Backend::portable && o.backend != Backend::automatic &&
         o.backend != Backend::esp32_s3)) return Status::invalid_argument;
    const size_t count = o.n_count == SIZE_MAX ? p.n - o.n_begin : o.n_count;
    if (count > p.n - o.n_begin) return Status::invalid_argument;
    const size_t elements = p.n * p.k;
    if (p.weight_bytes < elements / 4 + (elements % 4 != 0) ||
        !span_fits(p.m, p.input_stride, pk, p.input_bytes) ||
        !span_fits(p.m, p.output_stride, p.n, p.output_count) ||
        !span_fits(p.m, p.multiplier_stride, 1, p.multiplier_count) ||
        p.output_count > max / sizeof(float) || p.multiplier_count > max / sizeof(float))
        return Status::invalid_argument;
    if (s.weight_bytes < needed || (s.input_copy && s.input_copy_bytes < pk))
        return Status::insufficient_scratch;
    for (size_t m = 0; m < p.m; ++m) {
        const float multiplier = p.multipliers[m * p.multiplier_stride];
        if (!std::isfinite(multiplier) || multiplier <= 0) return Status::invalid_argument;
    }
    // Scan the owned partition before writing outputs, including non-byte row starts.
    for (size_t n = o.n_begin; n < o.n_begin + count; ++n)
        for (size_t k = 0; k < p.k; ++k)
            if (code_at(p.weights, n * p.k + k) == 3) return Status::invalid_weight;
    const bool direct = aligned16(p.input) && (p.m == 1 || p.input_stride % 16 == 0);
    const bool can_s3 = esp32_s3_available() && aligned16(s.weights) &&
        (direct || (s.input_copy && aligned16(s.input_copy)));
    if (o.backend == Backend::esp32_s3 && !can_s3)
        return esp32_s3_available() ? Status::invalid_argument : Status::unsupported;
    const bool use_s3 = o.backend != Backend::portable && can_s3;
    const size_t end = o.n_begin + count;
    for (size_t first = o.n_begin; first < end;) {
        const size_t rows = end - first < o.tile_rows ? end - first : o.tile_rows;
        for (size_t r = 0; r < rows; ++r) {
            int8_t* w = s.weights + r * pk;
            const size_t row_idx = first + r;
            const size_t start_bit_idx = row_idx * p.k;
            if (start_bit_idx % 4 == 0) {
                const uint8_t* row_bytes = p.weights + (start_bit_idx / 4);
                const size_t k4 = p.k / 4;
                for (size_t i = 0; i < k4; ++i) {
                    const uint32_t unpacked4 = s_unpack_lut[row_bytes[i]];
                    std::memcpy(w + (i * 4), &unpacked4, 4);
                }
                for (size_t k = k4 * 4; k < p.k; ++k) {
                    w[k] = static_cast<int8_t>(int(code_at(p.weights, start_bit_idx + k)) - 1);
                }
            } else {
                for (size_t k = 0; k < p.k; ++k)
                    w[k] = static_cast<int8_t>(int(code_at(p.weights, start_bit_idx + k)) - 1);
            }
            std::memset(w + p.k, 0, pk - p.k);
        }
        for (size_t m = 0; m < p.m; ++m) {
            const int8_t* x = p.input + m * p.input_stride;
            if (use_s3 && !direct) {
                std::memcpy(s.input_copy, x, p.k);
                std::memset(s.input_copy + p.k, 0, pk - p.k);
                x = s.input_copy;
            }
            const float factor = p.weight_scale / p.multipliers[m * p.multiplier_stride];
            for (size_t r = 0; r < rows; ++r) {
                int32_t sum;
                if (use_s3) {
                    const Status status = dot_s3_aligned(x, s.weights + r * pk, pk, &sum);
                    if (status != Status::ok) return status;
                } else {
                    sum = dot_portable(x, s.weights + r * pk, p.k);
                }
                p.output[m * p.output_stride + first + r] = float(sum) * factor;
            }
        }
        first += rows;
    }
    return Status::ok;
}
} // namespace kernels
} // namespace ruvi
