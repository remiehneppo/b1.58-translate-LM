#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

using namespace ruvi::kernels;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

static int8_t* align16(std::vector<int8_t>& v) {
    return reinterpret_cast<int8_t*>((reinterpret_cast<uintptr_t>(v.data()) + 15) & ~uintptr_t(15));
}

static void exercise(size_t k, size_t m, size_t n, size_t tile, int extreme, std::mt19937& rng) {
    const size_t pk = padded_k(k), xs = pk + 7, ys = n + 5;
    std::vector<int8_t> weights(n * k), xstorage(m * xs + 17);
    int8_t* x = align16(xstorage) + 1; // deliberately unaligned; strided microcopy path
    std::vector<uint8_t> packed((n * k + 3) / 4, 0);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = extreme ? int8_t(extreme > 0 ? 1 : -1) : int8_t(int(rng() % 3) - 1);
        packed[i / 4] |= uint8_t(uint8_t(weights[i] + 1) << (2 * (i % 4)));
    }
    for (size_t token = 0; token < m; ++token) {
        for (size_t j = 0; j < k; ++j)
            x[token * xs + j] = extreme ? int8_t(extreme > 0 ? -128 : 127) : int8_t(int(rng() % 256) - 128);
        std::fill(x + token * xs + k, x + token * xs + pk, int8_t(113));
    }
    std::vector<float> multipliers(m * 2, -1), y(m * ys, 12345), reference(m * ys, 12345);
    for (size_t token = 0; token < m; ++token) {
        multipliers[token * 2] = float(token + 1) * 0.625f;
        for (size_t row = 0; row < n; ++row) {
            int64_t acc = 0;
            for (size_t j = 0; j < k; ++j) acc += int64_t(weights[row * k + j]) * x[token * xs + j];
            reference[token * ys + row] = float(acc) * (0.375f / multipliers[token * 2]);
        }
    }
    std::vector<int8_t> tile_storage(weight_scratch_bytes(k, tile) + 16), copy_storage(pk + 16);
    Scratch scratch{align16(tile_storage), weight_scratch_bytes(k, tile), align16(copy_storage), pk};
    Problem p;
    p.weights = packed.data(); p.weight_bytes = packed.size();
    p.m = m; p.n = n; p.k = k; p.weight_scale = 0.375f;
    p.input = x; p.input_bytes = m * xs; p.input_stride = xs;
    p.multipliers = multipliers.data(); p.multiplier_count = multipliers.size(); p.multiplier_stride = 2;
    p.output = y.data(); p.output_count = y.size(); p.output_stride = ys;
    Options o; o.tile_rows = tile; o.backend = Backend::portable;
    CHECK(run(p, o, scratch) == Status::ok);
    CHECK(y == reference);
    std::fill(y.begin(), y.end(), 12345);
    o.backend = Backend::automatic;
    o.n_count = n / 2;
    CHECK(run(p, o, scratch) == Status::ok);
    // Use separate caller-owned tile storage for the second N partition.
    std::vector<int8_t> tile2_storage(scratch.weight_bytes + 16);
    Scratch scratch2{align16(tile2_storage), scratch.weight_bytes, scratch.input_copy, pk};
    o.n_begin = n / 2; o.n_count = SIZE_MAX;
    CHECK(run(p, o, scratch2) == Status::ok);
    CHECK(y == reference);
    if (esp32_s3_available()) {
        std::fill(y.begin(), y.end(), 12345);
        o.n_begin = 0; o.backend = Backend::esp32_s3;
        CHECK(run(p, o, scratch) == Status::ok);
        CHECK(y == reference);
    }
    if (rv1103_neon_available()) {
        std::fill(y.begin(), y.end(), 12345);
        o.n_begin = 0; o.backend = Backend::rv1103_neon;
        CHECK(run(p, o, scratch) == Status::ok);
        CHECK(y == reference);
    }
    // Also cover the aligned direct-input path, without optional microcopy.
    std::vector<int8_t> direct_storage(m * pk + 16);
    int8_t* direct = align16(direct_storage);
    for (size_t token = 0; token < m; ++token)
        std::copy(x + token * xs, x + token * xs + pk, direct + token * pk);
    const int8_t* original_input = p.input;
    p.input = direct; p.input_stride = pk; p.input_bytes = m * pk;
    Scratch direct_scratch{scratch.weights, scratch.weight_bytes, nullptr, 0};
    o = Options{}; o.tile_rows = tile;
    CHECK(run(p, o, direct_scratch) == Status::ok);
    CHECK(y == reference);
    o.backend = Backend::esp32_s3;
    CHECK(run(p, o, direct_scratch) == (esp32_s3_available() ? Status::ok : Status::unsupported));
    o.backend = Backend::rv1103_neon;
    CHECK(run(p, o, direct_scratch) == (rv1103_neon_available() ? Status::ok : Status::unsupported));
    p.input = original_input; p.input_stride = xs; p.input_bytes = m * xs;
    o = Options{}; o.tile_rows = tile;
    std::fill(y.begin(), y.end(), 12345);
    scratch.weight_bytes--;
    CHECK(run(p, o, scratch) == Status::insufficient_scratch);
    scratch.weight_bytes++;
    packed.back() |= uint8_t(3u << (2 * ((n * k - 1) % 4)));
    CHECK(run(p, o, scratch) == Status::invalid_weight);
    CHECK(std::all_of(y.begin(), y.end(), [](float f) { return f == 12345; }));
    packed.back() &= uint8_t(~(3u << (2 * ((n * k - 1) % 4))));
    multipliers.back() = 0; // untouched interleaved slot must be ignored
    multipliers[0] = 0;
    CHECK(run(p, o, scratch) == Status::invalid_argument);
    multipliers[0] = std::numeric_limits<float>::infinity();
    CHECK(run(p, o, scratch) == Status::invalid_argument);
    multipliers[0] = 1;
    p.input_bytes = pk - 1;
    CHECK(run(p, o, scratch) == Status::invalid_argument);
    p.input_bytes = m * xs;
    p.output_count = n - 1;
    CHECK(run(p, o, scratch) == Status::invalid_argument);
    p.output_count = y.size();
    p.k = 32769;
    CHECK(run(p, o, scratch) == Status::invalid_argument);
}

int main() {
    std::mt19937 rng(0x52555649);
    size_t cases = 0;
    for (size_t tile : {8u, 16u, 32u}) {
        for (size_t k : {1u, 3u, 5u, 15u, 16u, 17u, 31u, 65u, 257u}) {
            exercise(k, 3, 37, tile, 0, rng); ++cases;
        }
        for (int extreme : {-1, 1}) {
            exercise(32768, 2, 3, tile, extreme, rng); ++cases;
            exercise(32767, 2, 3, tile, extreme, rng); ++cases;
        }
        for (int i = 0; i < 40; ++i) {
            exercise(1 + rng() % 513, 1 + rng() % 6, 1 + rng() % 65, tile, 0, rng); ++cases;
        }
    }
    CHECK(weight_scratch_bytes(17, 8) == 256);
    CHECK(weight_scratch_bytes(0, 8) == 0);
    CHECK(weight_scratch_bytes(17, 9) == 0);
    alignas(16) int8_t a[16] = {}, b[16] = {};
    int32_t result = 7;
    CHECK(dot_s3_aligned(a + 1, b, 16, &result) == Status::invalid_argument);
    CHECK(dot_s3_aligned(a, b, 15, &result) == Status::invalid_argument);
    CHECK(dot_s3_aligned(a, b, 32784, &result) == Status::invalid_argument);
    CHECK(result == 7);
    CHECK(dot_s3_aligned(a, b, 16, &result) == (esp32_s3_available() ? Status::ok : Status::unsupported));
    CHECK(dot_rv1103_aligned(a + 1, b, 16, &result) == Status::invalid_argument);
    CHECK(dot_rv1103_aligned(a, b, 15, &result) == Status::invalid_argument);
    CHECK(dot_rv1103_aligned(a, b, 32784, &result) == Status::invalid_argument);
    CHECK(result == 7);
    CHECK(dot_rv1103_aligned(a, b, 16, &result) == (rv1103_neon_available() ? Status::ok : Status::unsupported));
    std::printf("PASS: %zu reference cases, tile rows 8/16/32, validation, S3 and RV1103 guard checks\n", cases);
}
