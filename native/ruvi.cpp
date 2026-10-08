#include "ruvi.h"
#include "kernels/kernel.h"
#include "parallel.h"
#include "tokenizer_data.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace ruvi {
namespace {
uint32_t u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
float f32(const uint8_t* p) {
    uint32_t bits = u32(p); float result; std::memcpy(&result, &bits, 4); return result;
}
uint32_t crc32(const uint8_t* data, size_t size, uint32_t previous = 0) {
    uint32_t crc = ~previous;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1)));
    }
    return ~crc;
}
int round_even(float x) {
    const float a = std::fabs(x), low = std::floor(a), part = a - low;
    int n = static_cast<int>(low);
    if (part > 0.5f || (part == 0.5f && (n & 1))) ++n;
    return x < 0 ? -n : n;
}
} // namespace

float Tensor::value(size_t index) const {
    if (kind == 0) return f32(data + 4 * index);
    if (kind == 3) return float(static_cast<int8_t>(data[index])) * scale;
    uint32_t bits = (uint32_t(data[2 * index]) | (uint32_t(data[2 * index + 1]) << 8)) << 16;
    float result; std::memcpy(&result, &bits, 4); return result;
}

bool Model::fail(const char* message) { error_ = message; position_ = 0; return false; }

bool Model::load(const void* image, size_t size) {
    image_bytes_ = 0; image_crc32_ = 0; capacity_ = 0; position_ = 0; layers_.clear();
    const auto* data = static_cast<const uint8_t*>(image);
    if (!data || size < 80 || std::memcmp(data, "RUVI0001", 8)) return fail("invalid native header/version");
    uint32_t* fields[] = {&cfg_.vocab, &cfg_.dim, &cfg_.layers, &cfg_.heads, &cfg_.kv_heads,
        &cfg_.head_dim, &cfg_.ff, &cfg_.max_context, &cfg_.eos, &cfg_.pad, &cfg_.tag_ru, &cfg_.tag_vi};
    for (size_t i = 0; i < 12; ++i) *fields[i] = u32(data + 8 + 4 * i);
    cfg_.theta = f32(data + 56); cfg_.eps = f32(data + 60);
    size_t length = u32(data + 64);
    if (length < 80 || length > size) return fail("truncated native image");
    if (!cfg_.vocab || cfg_.vocab > 65536 || !cfg_.dim || cfg_.dim > 32768
        || !cfg_.ff || cfg_.ff > 32768 || cfg_.layers > 64 || !cfg_.max_context || cfg_.max_context > 65536
        || !cfg_.heads || !cfg_.kv_heads || cfg_.heads % cfg_.kv_heads || cfg_.heads > 1024
        || !cfg_.head_dim || cfg_.head_dim > 1024 || cfg_.head_dim % 2
        || uint64_t(cfg_.heads) * cfg_.head_dim > 32768
        || cfg_.eos >= cfg_.vocab || cfg_.pad >= cfg_.vocab || cfg_.tag_ru >= cfg_.vocab || cfg_.tag_vi >= cfg_.vocab
        || !std::isfinite(cfg_.theta) || cfg_.theta <= 0 || !std::isfinite(cfg_.eps) || cfg_.eps <= 0)
        return fail("invalid model dimensions/config");
    uint8_t header[80]; std::memcpy(header, data, 80); std::memset(header + 68, 0, 4);
    if (crc32(data + 80, length - 80, crc32(header, 80)) != u32(data + 68)) return fail("native image CRC mismatch");
    size_t offset = 80;
    auto tensor = [&](Tensor& t, uint32_t rows, uint32_t cols, bool ternary) {
        if (offset > length || length - offset < 32) return false;
        t.rows = u32(data + offset); t.cols = u32(data + offset + 4); t.kind = u32(data + offset + 8);
        t.scale = f32(data + offset + 12);
        uint64_t bytes = u32(data + offset + 16), elements = uint64_t(rows) * cols;
        uint64_t expected = t.kind == 2 ? (elements + 3) / 4 : elements * (t.kind == 1 ? 2 : (t.kind == 3 ? 1 : 4));
        if (t.rows != rows || t.cols != cols || t.kind > 3 || (t.kind == 2 && !ternary)
            || bytes != expected || !std::isfinite(t.scale) || t.scale <= 0) return false;
        offset += 32;
        uint64_t aligned = (bytes + 15) & ~uint64_t(15);
        if (aligned > length - offset) return false;
        t.data = data + offset;
        if (t.kind == 2) {
            for (uint64_t i = 0; i < elements; ++i)
                if (((t.data[i / 4] >> (2 * (i % 4))) & 3) == 3) return false;
        } else if (t.kind == 3) {
            // int8 elements are valid
        } else {
            for (uint64_t i = 0; i < elements; ++i) if (!std::isfinite(t.value(i))) return false;
        }
        offset += static_cast<size_t>(aligned);
        return true;
    };
    const uint32_t qdim = cfg_.heads * cfg_.head_dim, kdim = cfg_.kv_heads * cfg_.head_dim;
    if (!tensor(embedding_, cfg_.vocab, cfg_.dim, false)) return fail("invalid embedding tensor");
    layers_.resize(cfg_.layers);
    for (auto& l : layers_) {
        if (!tensor(l.attn_norm, 1, cfg_.dim, false) || !tensor(l.q_norm, 1, qdim, false)
            || !tensor(l.k_norm, 1, kdim, false) || !tensor(l.q, qdim, cfg_.dim, true)
            || !tensor(l.k, kdim, cfg_.dim, true) || !tensor(l.v, kdim, cfg_.dim, true)
            || !tensor(l.o, cfg_.dim, qdim, true) || !tensor(l.ffn_norm, 1, cfg_.dim, false)
            || !tensor(l.fc1, cfg_.ff, cfg_.dim, true) || !tensor(l.fc2, cfg_.dim, cfg_.ff, true))
            return fail("invalid layer tensor");
    }
    if (!tensor(norm_, 1, cfg_.dim, false) || offset != length) return fail("invalid final norm/image length");
    image_bytes_ = length; image_crc32_ = u32(data + 68); error_ = nullptr;
    set_options(options_);
    return true;
}

size_t Model::scratch_bytes() const {
    if (!image_bytes_) return 0;
    const uint32_t qdim = cfg_.heads * cfg_.head_dim;
    const size_t max_k = std::max({size_t(cfg_.dim), size_t(qdim), size_t(cfg_.ff)});
    const size_t pk = kernels::padded_k(max_k);
    const size_t tile_w = kernels::weight_scratch_bytes(max_k, options_.tile_rows);
    return (options_.cores > 1 ? 2 : 1) * tile_w + pk + 64;
}

void Model::set_options(const KernelOptions& options) {
    options_ = options;
    if (options_.backend == Backend::Scalar) {
        effective_backend_ = Backend::Scalar;
    } else if (options_.backend == Backend::S3) {
        effective_backend_ = kernels::esp32_s3_available() ? Backend::S3 : Backend::Tiled;
    } else {
        effective_backend_ = Backend::Tiled;
    }
}

bool Model::bind_scratch(void* scratch, size_t bytes) {
    scratch_mem_ = scratch;
    scratch_bytes_ = bytes;
    tile_scratch_[0] = nullptr;
    tile_scratch_[1] = nullptr;
    quant_scratch_ = nullptr;
    if (!scratch || bytes < scratch_bytes()) return false;
    uintptr_t addr = reinterpret_cast<uintptr_t>(scratch);
    uintptr_t aligned = (addr + 15u) & ~uintptr_t(15u);
    const uint32_t qdim = cfg_.heads * cfg_.head_dim;
    const size_t max_k = std::max({size_t(cfg_.dim), size_t(qdim), size_t(cfg_.ff)});
    const size_t pk = kernels::padded_k(max_k);
    const size_t tile_w = kernels::weight_scratch_bytes(max_k, options_.tile_rows);
    int8_t* p = reinterpret_cast<int8_t*>(aligned);
    tile_scratch_[0] = p; p += tile_w;
    tile_scratch_bytes_ = tile_w;
    if (options_.cores > 1) {
        tile_scratch_[1] = p; p += tile_w;
    }
    quant_scratch_ = p;
    quant_scratch_bytes_ = pk;
    return true;
}

size_t Model::workspace_bytes(uint32_t context) const {
    if (!image_bytes_ || !context || context > cfg_.max_context) return 0;
    const uint64_t qdim = uint64_t(cfg_.heads) * cfg_.head_dim, kdim = uint64_t(cfg_.kv_heads) * cfg_.head_dim;
    const uint64_t floats = 2 * uint64_t(cfg_.layers) * context * kdim + uint64_t(context) * cfg_.dim
        + 2 * cfg_.dim + 2 * qdim + cfg_.ff + context + cfg_.vocab + 2 * uint64_t(context) * (cfg_.head_dim / 2);
    const size_t max_dim = std::max({size_t(cfg_.dim), size_t(qdim), size_t(cfg_.ff)});
    const size_t pk = kernels::padded_k(max_dim);
    const size_t fallback_scratch = scratch_bytes();
    uint64_t bytes = floats * 4 + pk + fallback_scratch + 64;
    return bytes > std::numeric_limits<size_t>::max() ? 0 : static_cast<size_t>(bytes);
}

bool Model::bind_workspace(void* memory, size_t bytes, uint32_t context) {
    capacity_ = 0;
    size_t required = workspace_bytes(context);
    if (!memory || reinterpret_cast<uintptr_t>(memory) % alignof(float) || !required || bytes < required)
        return fail("invalid/insufficient workspace or context");
    capacity_ = context; position_ = 0;
    float* next = static_cast<float*>(memory);
    auto take = [&](size_t count) { float* result = next; next += count; return result; };
    const size_t kd = size_t(cfg_.kv_heads) * cfg_.head_dim;
    cache_ = take(size_t(2) * cfg_.layers * context * kd);
    hidden_ = take(size_t(context) * cfg_.dim);
    normed_ = take(cfg_.dim); q_ = take(cfg_.heads * cfg_.head_dim);
    attended_ = take(cfg_.heads * cfg_.head_dim); ff_ = take(cfg_.ff); temp_ = take(cfg_.dim);
    scores_ = take(context); logits_ = take(cfg_.vocab);
    rope_cos_ = take(size_t(context) * (cfg_.head_dim / 2));
    rope_sin_ = take(size_t(context) * (cfg_.head_dim / 2));

    // Precompute RoPE table for fast continuous position encoding
    for (uint32_t i = 0; i < cfg_.head_dim; i += 2) {
        const float freq = 1.0f / std::pow(cfg_.theta, float(i) / cfg_.head_dim);
        for (uint32_t pos = 0; pos < context; ++pos) {
            const float angle = float(pos) * freq;
            rope_cos_[size_t(pos) * (cfg_.head_dim / 2) + i / 2] = std::cos(angle);
            rope_sin_[size_t(pos) * (cfg_.head_dim / 2) + i / 2] = std::sin(angle);
        }
    }

    uintptr_t raw_p = reinterpret_cast<uintptr_t>(next);
    uintptr_t aligned_p = (raw_p + 15u) & ~uintptr_t(15u);
    quantized_ = reinterpret_cast<int8_t*>(aligned_p);

    const size_t max_dim = std::max({size_t(cfg_.dim), size_t(cfg_.heads * cfg_.head_dim), size_t(cfg_.ff)});
    const size_t pk = kernels::padded_k(max_dim);
    int8_t* after_quant = quantized_ + pk;

    // If caller has not bound a dedicated fast SRAM scratch, carve from workspace
    if (!scratch_mem_) {
        uintptr_t s_aligned = (reinterpret_cast<uintptr_t>(after_quant) + 15u) & ~uintptr_t(15u);
        bind_scratch(reinterpret_cast<void*>(s_aligned), scratch_bytes());
    }

    error_ = nullptr; return true;
}

void Model::rms(const float* input, float* output, const Tensor& weight) {
    float sum = 0;
    for (uint32_t i = 0; i < weight.cols; ++i) sum += input[i] * input[i];
    const float inverse = 1.0f / std::sqrt(sum / weight.cols + cfg_.eps);
    for (uint32_t i = 0; i < weight.cols; ++i) output[i] = input[i] * inverse * weight.value(i);
}

void Model::quantize_row(const float* input, uint32_t cols, int8_t* out, float& multiplier) {
    float maximum = 1e-5f;
    for (uint32_t i = 0; i < cols; ++i) {
        maximum = std::max(maximum, std::fabs(input[i]));
    }
    multiplier = 127.0f / maximum;
    for (uint32_t i = 0; i < cols; ++i) {
        out[i] = static_cast<int8_t>(std::max(-128, std::min(127, round_even(input[i] * multiplier))));
    }
    const size_t pk = kernels::padded_k(cols);
    if (pk > cols) {
        std::memset(out + cols, 0, pk - cols);
    }
}

void Model::linear_ternary(const Tensor& weight, const int8_t* input_quant, float multiplier, float* output,
                           uint32_t tokens, size_t in_stride, size_t out_stride) {
    const size_t pk = kernels::padded_k(weight.cols);
    const size_t stride_x = in_stride ? in_stride : pk;
    const size_t stride_y = out_stride ? out_stride : weight.rows;

    if (effective_backend_ != Backend::Scalar && tile_scratch_[0]) {
        kernels::Problem prob;
        prob.weights = weight.data;
        prob.weight_bytes = (size_t(weight.rows) * weight.cols + 3) / 4;
        prob.m = tokens;
        prob.n = weight.rows;
        prob.k = weight.cols;
        prob.weight_scale = weight.scale;
        prob.input = input_quant;
        prob.input_bytes = tokens * stride_x;
        prob.input_stride = stride_x;
        prob.multipliers = &multiplier;
        prob.multiplier_count = tokens;
        prob.multiplier_stride = in_stride ? 1 : 0;
        prob.output = output;
        prob.output_count = tokens * stride_y;
        prob.output_stride = stride_y;

        kernels::Options opts;
        opts.tile_rows = options_.tile_rows;
        opts.backend = (effective_backend_ == Backend::S3) ? kernels::Backend::esp32_s3 : kernels::Backend::portable;

        if (options_.cores > 1 && tile_scratch_[1] && weight.rows >= 32) {
            struct ParallelTernaryJob {
                const kernels::Problem* p;
                kernels::Options opts;
                int8_t* tile[2];
                size_t tile_bytes;
                int8_t* quant_scratch;
                size_t quant_bytes;
                bool ok[2];
            } job;
            job.p = &prob;
            job.opts = opts;
            job.tile[0] = tile_scratch_[0];
            job.tile[1] = tile_scratch_[1];
            job.tile_bytes = tile_scratch_bytes_;
            job.quant_scratch = quant_scratch_;
            job.quant_bytes = quant_scratch_bytes_;
            job.ok[0] = false;
            job.ok[1] = false;

            parallel_run([](void* arg, uint32_t part) {
                auto* j = static_cast<ParallelTernaryJob*>(arg);
                const size_t total_n = j->p->n;
                const size_t n0 = (part * total_n) / 2;
                const size_t n1 = ((part + 1) * total_n) / 2;
                kernels::Options o = j->opts;
                o.n_begin = n0;
                o.n_count = n1 - n0;
                kernels::Scratch scr;
                scr.weights = j->tile[part];
                scr.weight_bytes = j->tile_bytes;
                scr.input_copy = (part == 0) ? j->quant_scratch : nullptr;
                scr.input_copy_bytes = (part == 0) ? j->quant_bytes : 0;
                j->ok[part] = (kernels::run(*(j->p), o, scr) == kernels::Status::ok);
            }, &job, 2);

            if (job.ok[0] && job.ok[1]) return;
        } else {
            kernels::Scratch scr;
            scr.weights = tile_scratch_[0];
            scr.weight_bytes = tile_scratch_bytes_;
            scr.input_copy = quant_scratch_;
            scr.input_copy_bytes = quant_scratch_bytes_;

            if (kernels::run(prob, opts, scr) == kernels::Status::ok) {
                return;
            }
        }
    }

    // Reference fallback
    for (uint32_t t = 0; t < tokens; ++t) {
        const int8_t* x = input_quant + t * stride_x;
        float* y = output + t * stride_y;
        const float mult = in_stride ? *(&multiplier + t) : multiplier;
        const float factor = weight.scale / mult;
        for (uint32_t row = 0; row < weight.rows; ++row) {
            int32_t sum = 0;
            const size_t start = size_t(row) * weight.cols;
            for (uint32_t col = 0; col < weight.cols; ++col) {
                const size_t index = start + col;
                const int code = (weight.data[index / 4] >> (2 * (index % 4))) & 3;
                if (code == 2) sum += x[col];
                else if (code == 0) sum -= x[col];
            }
            y[row] = float(sum) * factor;
        }
    }
}

namespace {
struct DenseJob {
    const Tensor* weight;
    const float* input;
    float* output;
    uint32_t rows;
    uint32_t cols;
    uint32_t parts;
};

void linear_dense_part(void* arg, uint32_t part) {
    const auto* j = static_cast<const DenseJob*>(arg);
    const uint32_t r0 = (part * j->rows) / j->parts;
    const uint32_t r1 = ((part + 1) * j->rows) / j->parts;
    const uint32_t cols = j->cols;

    if (j->weight->kind == 1) { // BF16 direct pointer arithmetic
        const uint16_t* d16 = reinterpret_cast<const uint16_t*>(j->weight->data);
        for (uint32_t r = r0; r < r1; ++r) {
            float sum = 0.0f;
            const uint16_t* row = d16 + size_t(r) * cols;
            for (uint32_t c = 0; c < cols; ++c) {
                uint32_t bits = uint32_t(row[c]) << 16;
                float w; std::memcpy(&w, &bits, 4);
                sum += j->input[c] * w;
            }
            j->output[r] = sum;
        }
    } else if (j->weight->kind == 0) { // FP32 direct pointer arithmetic
        const float* d32 = reinterpret_cast<const float*>(j->weight->data);
        for (uint32_t r = r0; r < r1; ++r) {
            float sum = 0.0f;
            const float* row = d32 + size_t(r) * cols;
            for (uint32_t c = 0; c < cols; ++c) sum += j->input[c] * row[c];
            j->output[r] = sum;
        }
    } else {
        for (uint32_t r = r0; r < r1; ++r) {
            float sum = 0.0f;
            const size_t start = size_t(r) * cols;
            for (uint32_t c = 0; c < cols; ++c) {
                sum += j->input[c] * j->weight->value(start + c);
            }
            j->output[r] = sum;
        }
    }
}

struct Int8Job {
    const Tensor* weight;
    const int8_t* input_quant;
    float factor;
    float* output;
    uint32_t rows;
    uint32_t cols;
    uint32_t parts;
    const uint8_t* mask;
    bool use_s3;
};

void linear_int8_part(void* arg, uint32_t part) {
    const auto* j = static_cast<const Int8Job*>(arg);
    const uint32_t r0 = (part * j->rows) / j->parts;
    const uint32_t r1 = ((part + 1) * j->rows) / j->parts;
    const uint32_t cols = j->cols;
    const int8_t* w_base = reinterpret_cast<const int8_t*>(j->weight->data);
    const float factor = j->factor;
    const uint8_t* mask = j->mask;

    for (uint32_t r = r0; r < r1; ++r) {
        if (mask && !((mask[r >> 3] >> (r & 7)) & 1)) {
            j->output[r] = -1e9f;
            continue;
        }
        const int8_t* w_row = w_base + size_t(r) * cols;
        int32_t sum = 0;
        if (j->use_s3) {
            kernels::dot_s3_aligned(j->input_quant, w_row, cols, &sum);
        } else {
            for (uint32_t c = 0; c < cols; ++c) {
                sum += int32_t(j->input_quant[c]) * int32_t(w_row[c]);
            }
        }
        j->output[r] = float(sum) * factor;
    }
}

inline void lookup_embedding(const Tensor& emb, uint32_t token, uint32_t dim, float* out) {
    if (emb.kind == 3) {
        const int8_t* p8 = reinterpret_cast<const int8_t*>(emb.data) + size_t(token) * dim;
        const float s = emb.scale;
        for (uint32_t d = 0; d < dim; ++d) out[d] = float(p8[d]) * s;
    } else if (emb.kind == 1) {
        const uint16_t* p16 = reinterpret_cast<const uint16_t*>(emb.data) + size_t(token) * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            uint32_t bits = uint32_t(p16[d]) << 16;
            float f; std::memcpy(&f, &bits, 4);
            out[d] = f;
        }
    } else if (emb.kind == 0) {
        const float* p32 = reinterpret_cast<const float*>(emb.data) + size_t(token) * dim;
        for (uint32_t d = 0; d < dim; ++d) out[d] = p32[d];
    } else {
        for (uint32_t d = 0; d < dim; ++d) out[d] = emb.value(size_t(token) * dim + d);
    }
}
} // namespace

void Model::linear_int8(const Tensor& weight, const int8_t* input_quant, float multiplier, float* output, const uint8_t* subvocab_mask) {
    const float factor = weight.scale / multiplier;
    const uint32_t parts = (options_.cores > 1 && weight.rows >= 64) ? 2 : 1;
    const bool use_s3 = (effective_backend_ == Backend::S3 && (weight.cols % 16 == 0));
    Int8Job job{&weight, input_quant, factor, output, weight.rows, weight.cols, parts, subvocab_mask, use_s3};
    parallel_run(linear_int8_part, &job, parts);
}

void Model::linear(const Tensor& weight, const float* input, float* output) {
    if (weight.kind == 2) {
        float multiplier = 1.0f;
        quantize_row(input, weight.cols, quantized_, multiplier);
        linear_ternary(weight, quantized_, multiplier, output, 1, 0, 0);
    } else if (weight.kind == 3) {
        float multiplier = 1.0f;
        quantize_row(input, weight.cols, quantized_, multiplier);
        linear_int8(weight, quantized_, multiplier, output, nullptr);
    } else {
        const uint32_t parts = (options_.cores > 1 && weight.rows >= 64) ? 2 : 1;
        DenseJob job{&weight, input, output, weight.rows, weight.cols, parts};
        parallel_run(linear_dense_part, &job, parts);
    }
}

void Model::rope(float* values, uint32_t heads, uint32_t position) {
    if (rope_cos_ && rope_sin_ && position < capacity_) {
        const float* cos_p = rope_cos_ + size_t(position) * (cfg_.head_dim / 2);
        const float* sin_p = rope_sin_ + size_t(position) * (cfg_.head_dim / 2);
        for (uint32_t h = 0; h < heads; ++h) {
            float* h_vals = values + h * cfg_.head_dim;
            for (uint32_t i = 0; i < cfg_.head_dim; i += 2) {
                const float c = cos_p[i / 2], s = sin_p[i / 2];
                const float x = h_vals[i], y = h_vals[i + 1];
                h_vals[i] = x * c - y * s;
                h_vals[i + 1] = x * s + y * c;
            }
        }
        return;
    }
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t i = 0; i < cfg_.head_dim; i += 2) {
            const float angle = position / std::pow(cfg_.theta, float(i) / cfg_.head_dim);
            const float c = std::cos(angle), s = std::sin(angle);
            float* pair = values + h * cfg_.head_dim + i;
            const float x = pair[0], y = pair[1]; pair[0] = x * c - y * s; pair[1] = x * s + y * c;
        }
    }
}

void Model::keys(const Layer& layer, uint32_t index, const float* input, uint32_t position) {
    const size_t kd = size_t(cfg_.kv_heads) * cfg_.head_dim, stride = size_t(capacity_) * kd;
    float* k = cache_ + 2 * index * stride + position * kd;
    float* v = cache_ + (2 * index + 1) * stride + position * kd;
    rms(input, normed_, layer.attn_norm);
    if (layer.k.kind == 2 && layer.v.kind == 2) {
        float multiplier = 1.0f;
        quantize_row(normed_, cfg_.dim, quantized_, multiplier);
        linear_ternary(layer.k, quantized_, multiplier, k);
        rms(k, k, layer.k_norm);
        rope(k, cfg_.kv_heads, position);
        linear_ternary(layer.v, quantized_, multiplier, v);
    } else {
        linear(layer.k, normed_, k); rms(k, k, layer.k_norm);
        rope(k, cfg_.kv_heads, position); linear(layer.v, normed_, v);
    }
}

void Model::block(const Layer& layer, uint32_t index, float* hidden, uint32_t position, uint32_t attend_count) {
    const size_t kd = size_t(cfg_.kv_heads) * cfg_.head_dim, stride = size_t(capacity_) * kd;
    const float* k = cache_ + 2 * index * stride; const float* v = k + stride;
    rms(hidden, normed_, layer.attn_norm);
    if (layer.q.kind == 2) {
        float multiplier = 1.0f;
        quantize_row(normed_, cfg_.dim, quantized_, multiplier);
        linear_ternary(layer.q, quantized_, multiplier, q_);
    } else {
        linear(layer.q, normed_, q_);
    }
    rms(q_, q_, layer.q_norm);
    rope(q_, cfg_.heads, position);

    const float factor = 1.0f / std::sqrt(float(cfg_.head_dim));
    for (uint32_t h = 0; h < cfg_.heads; ++h) {
        const uint32_t kh = h / (cfg_.heads / cfg_.kv_heads);
        const float* query = q_ + h * cfg_.head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t t = 0; t < attend_count; ++t) {
            float dot = 0; const float* key = k + t * kd + kh * cfg_.head_dim;
            for (uint32_t d = 0; d < cfg_.head_dim; ++d) dot += query[d] * key[d];
            scores_[t] = dot * factor; maximum = std::max(maximum, scores_[t]);
        }
        float total = 0;
        for (uint32_t t = 0; t < attend_count; ++t) { scores_[t] = std::exp(scores_[t] - maximum); total += scores_[t]; }
        const float inv_total = total > 0 ? 1.0f / total : 0.0f;
        float* att_head = attended_ + h * cfg_.head_dim;
        std::memset(att_head, 0, cfg_.head_dim * sizeof(float));
        for (uint32_t t = 0; t < attend_count; ++t) {
            const float weight_val = scores_[t] * inv_total;
            if (weight_val == 0.0f) continue;
            const float* v_t = v + t * kd + kh * cfg_.head_dim;
            for (uint32_t d = 0; d < cfg_.head_dim; ++d) {
                att_head[d] += weight_val * v_t[d];
            }
        }
    }
    linear(layer.o, attended_, temp_);
    for (uint32_t d = 0; d < cfg_.dim; ++d) hidden[d] += temp_[d];
    rms(hidden, normed_, layer.ffn_norm); linear(layer.fc1, normed_, ff_);
    for (uint32_t d = 0; d < cfg_.ff; ++d) { const float x = std::max(0.0f, ff_[d]); ff_[d] = x * x; }
    linear(layer.fc2, ff_, temp_);
    for (uint32_t d = 0; d < cfg_.dim; ++d) hidden[d] += temp_[d];
}

bool Model::output(const float* hidden) {
    rms(hidden, normed_, norm_);
    const uint8_t* mask = nullptr;
    if (options_.subvocab && cfg_.vocab == tokenizer_data::VOCAB_SIZE) {
        if (target_lang_ == cfg_.tag_ru) {
            mask = tokenizer_data::SUBVOCAB_RU_MASK;
        } else if (target_lang_ == cfg_.tag_vi) {
            mask = tokenizer_data::SUBVOCAB_VI_MASK;
        }
    }
    if (embedding_.kind == 3) {
        float multiplier = 1.0f;
        quantize_row(normed_, embedding_.cols, quantized_, multiplier);
        linear_int8(embedding_, quantized_, multiplier, logits_, mask);
    } else {
        linear(embedding_, normed_, logits_);
        if (mask) {
            for (uint32_t i = 0; i < cfg_.vocab; ++i) {
                if (!((mask[i >> 3] >> (i & 7)) & 1)) logits_[i] = -1e9f;
            }
        }
    }
    for (uint32_t i = 0; i < cfg_.vocab; ++i) if (!std::isfinite(logits_[i])) return fail("nonfinite inference logits");
    error_ = nullptr; return true;
}

bool Model::prefill(const uint32_t* ids, uint32_t count) {
    position_ = 0; error_ = nullptr;
    if (!capacity_ || !ids || !count || count > capacity_) return fail("invalid prompt/workspace");
    target_lang_ = ids[0];
    for (uint32_t t = 0; t < count; ++t) {
        if (ids[t] >= cfg_.vocab) return fail("prompt token outside vocabulary");
        lookup_embedding(embedding_, ids[t], cfg_.dim, hidden_ + size_t(t) * cfg_.dim);
    }
    for (uint32_t i = 0; i < cfg_.layers; ++i) {
        for (uint32_t t = 0; t < count; ++t) {
            keys(layers_[i], i, hidden_ + size_t(t) * cfg_.dim, t);
            if (error_) return false;
        }
        // All prefix keys exist before updating any hidden row: bidirectional Prefix-LM.
        for (uint32_t t = 0; t < count; ++t) {
            block(layers_[i], i, hidden_ + size_t(t) * cfg_.dim, t, count);
            if (error_) return false;
        }
    }
    position_ = count; return output(hidden_ + size_t(count - 1) * cfg_.dim);
}

bool Model::decode(uint32_t token) {
    if (!position_ || position_ >= capacity_ || token >= cfg_.vocab) return fail("decode token/context invalid");
    lookup_embedding(embedding_, token, cfg_.dim, hidden_);
    const size_t kd = size_t(cfg_.kv_heads) * cfg_.head_dim, stride = size_t(capacity_) * kd;

    for (uint32_t i = 0; i < cfg_.layers; ++i) {
        const auto& layer = layers_[i];
        float* k = cache_ + 2 * i * stride + position_ * kd;
        float* v = cache_ + (2 * i + 1) * stride + position_ * kd;
        rms(hidden_, normed_, layer.attn_norm);

        if (layer.q.kind == 2 && layer.k.kind == 2 && layer.v.kind == 2) {
            float multiplier = 1.0f;
            quantize_row(normed_, cfg_.dim, quantized_, multiplier);
            linear_ternary(layer.k, quantized_, multiplier, k);
            linear_ternary(layer.v, quantized_, multiplier, v);
            linear_ternary(layer.q, quantized_, multiplier, q_);
        } else {
            linear(layer.k, normed_, k);
            linear(layer.v, normed_, v);
            linear(layer.q, normed_, q_);
        }
        rms(k, k, layer.k_norm);
        rope(k, cfg_.kv_heads, position_);
        rms(q_, q_, layer.q_norm);
        rope(q_, cfg_.heads, position_);

        const float factor = 1.0f / std::sqrt(float(cfg_.head_dim));
        const uint32_t attend_count = position_ + 1;
        const float* k_base = cache_ + 2 * i * stride;
        const float* v_base = k_base + stride;

        for (uint32_t h = 0; h < cfg_.heads; ++h) {
            const uint32_t kh = h / (cfg_.heads / cfg_.kv_heads);
            const float* query = q_ + h * cfg_.head_dim;
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t t = 0; t < attend_count; ++t) {
                float dot = 0; const float* key = k_base + t * kd + kh * cfg_.head_dim;
                for (uint32_t d = 0; d < cfg_.head_dim; ++d) dot += query[d] * key[d];
                scores_[t] = dot * factor; maximum = std::max(maximum, scores_[t]);
            }
            float total = 0;
            for (uint32_t t = 0; t < attend_count; ++t) { scores_[t] = std::exp(scores_[t] - maximum); total += scores_[t]; }
            const float inv_total = total > 0 ? 1.0f / total : 0.0f;
            float* att_head = attended_ + h * cfg_.head_dim;
            std::memset(att_head, 0, cfg_.head_dim * sizeof(float));
            for (uint32_t t = 0; t < attend_count; ++t) {
                const float weight_val = scores_[t] * inv_total;
                if (weight_val == 0.0f) continue;
                const float* v_t = v_base + t * kd + kh * cfg_.head_dim;
                for (uint32_t d = 0; d < cfg_.head_dim; ++d) {
                    att_head[d] += weight_val * v_t[d];
                }
            }
        }
        linear(layer.o, attended_, temp_);
        for (uint32_t d = 0; d < cfg_.dim; ++d) hidden_[d] += temp_[d];
        rms(hidden_, normed_, layer.ffn_norm); linear(layer.fc1, normed_, ff_);
        for (uint32_t d = 0; d < cfg_.ff; ++d) { const float x = std::max(0.0f, ff_[d]); ff_[d] = x * x; }
        linear(layer.fc2, ff_, temp_);
        for (uint32_t d = 0; d < cfg_.dim; ++d) hidden_[d] += temp_[d];
    }
    ++position_; return output(hidden_);
}

uint32_t Model::argmax() const {
    if (!position_ || error_) return cfg_.pad;
    return static_cast<uint32_t>(std::max_element(logits_, logits_ + cfg_.vocab) - logits_);
}
} // namespace ruvi
