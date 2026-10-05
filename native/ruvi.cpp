#include "ruvi.h"
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
}
float Tensor::value(size_t index) const {
    if (kind == 0) return f32(data + 4 * index);
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
        uint64_t expected = t.kind == 2 ? (elements + 3) / 4 : elements * (t.kind == 1 ? 2 : 4);
        if (t.rows != rows || t.cols != cols || t.kind > 2 || (t.kind == 2 && !ternary)
            || bytes != expected || !std::isfinite(t.scale) || t.scale <= 0) return false;
        offset += 32;
        uint64_t aligned = (bytes + 15) & ~uint64_t(15);
        if (aligned > length - offset) return false;
        t.data = data + offset;
        if (t.kind == 2) {
            for (uint64_t i = 0; i < elements; ++i)
                if (((t.data[i / 4] >> (2 * (i % 4))) & 3) == 3) return false;
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
    image_bytes_ = length; image_crc32_ = u32(data + 68); error_ = nullptr; return true;
}
size_t Model::workspace_bytes(uint32_t context) const {
    if (!image_bytes_ || !context || context > cfg_.max_context) return 0;
    uint64_t qdim = uint64_t(cfg_.heads) * cfg_.head_dim, kdim = uint64_t(cfg_.kv_heads) * cfg_.head_dim;
    uint64_t floats = 2 * uint64_t(cfg_.layers) * context * kdim + uint64_t(context) * cfg_.dim
        + 2 * cfg_.dim + 2 * qdim + cfg_.ff + context + cfg_.vocab;
    uint64_t bytes = floats * 4 + std::max({uint64_t(cfg_.dim), qdim, uint64_t(cfg_.ff)});
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
    cache_ = take(size_t(2) * cfg_.layers * context * cfg_.kv_heads * cfg_.head_dim);
    hidden_ = take(size_t(context) * cfg_.dim);
    normed_ = take(cfg_.dim); q_ = take(cfg_.heads * cfg_.head_dim);
    attended_ = take(cfg_.heads * cfg_.head_dim); ff_ = take(cfg_.ff); temp_ = take(cfg_.dim);
    scores_ = take(context); logits_ = take(cfg_.vocab); quantized_ = reinterpret_cast<int8_t*>(next);
    error_ = nullptr; return true;
}
void Model::rms(const float* input, float* output, const Tensor& weight) {
    float sum = 0;
    for (uint32_t i = 0; i < weight.cols; ++i) sum += input[i] * input[i];
    const float inverse = 1.0f / std::sqrt(sum / weight.cols + cfg_.eps);
    for (uint32_t i = 0; i < weight.cols; ++i) output[i] = input[i] * inverse * weight.value(i);
}
void Model::linear(const Tensor& weight, const float* input, float* output) {
    if (weight.kind == 2) {
        float maximum = 1e-5f;
        for (uint32_t i = 0; i < weight.cols; ++i) {
            if (!std::isfinite(input[i])) { fail("nonfinite quantization input"); return; }
            maximum = std::max(maximum, std::fabs(input[i]));
        }
        const float multiplier = 127.0f / maximum;
        for (uint32_t i = 0; i < weight.cols; ++i)
            quantized_[i] = static_cast<int8_t>(std::max(-128, std::min(127, round_even(input[i] * multiplier))));
        for (uint32_t row = 0; row < weight.rows; ++row) {
            int32_t sum = 0;
            const size_t start = size_t(row) * weight.cols;
            for (uint32_t col = 0; col < weight.cols; ++col) {
                const size_t index = start + col;
                const int code = (weight.data[index / 4] >> (2 * (index % 4))) & 3;
                if (code == 2) sum += quantized_[col];
                else if (code == 0) sum -= quantized_[col];
            }
            output[row] = sum * (weight.scale / multiplier);
        }
    } else {
        for (uint32_t row = 0; row < weight.rows; ++row) {
            float sum = 0;
            const size_t start = size_t(row) * weight.cols;
            for (uint32_t col = 0; col < weight.cols; ++col) sum += input[col] * weight.value(start + col);
            output[row] = sum;
        }
    }
}
void Model::rope(float* values, uint32_t heads, uint32_t position) {
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
    rms(input, normed_, layer.attn_norm); linear(layer.k, normed_, k); rms(k, k, layer.k_norm);
    rope(k, cfg_.kv_heads, position); linear(layer.v, normed_, v);
}
void Model::block(const Layer& layer, uint32_t index, float* hidden, uint32_t position, uint32_t attend_count) {
    const size_t kd = size_t(cfg_.kv_heads) * cfg_.head_dim, stride = size_t(capacity_) * kd;
    const float* k = cache_ + 2 * index * stride; const float* v = k + stride;
    rms(hidden, normed_, layer.attn_norm); linear(layer.q, normed_, q_); rms(q_, q_, layer.q_norm);
    rope(q_, cfg_.heads, position);
    const float factor = 1.0f / std::sqrt(float(cfg_.head_dim));
    for (uint32_t h = 0; h < cfg_.heads; ++h) {
        uint32_t kh = h / (cfg_.heads / cfg_.kv_heads);
        const float* query = q_ + h * cfg_.head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t t = 0; t < attend_count; ++t) {
            float dot = 0; const float* key = k + t * kd + kh * cfg_.head_dim;
            for (uint32_t d = 0; d < cfg_.head_dim; ++d) dot += query[d] * key[d];
            scores_[t] = dot * factor; maximum = std::max(maximum, scores_[t]);
        }
        float total = 0;
        for (uint32_t t = 0; t < attend_count; ++t) { scores_[t] = std::exp(scores_[t] - maximum); total += scores_[t]; }
        for (uint32_t d = 0; d < cfg_.head_dim; ++d) {
            float sum = 0;
            for (uint32_t t = 0; t < attend_count; ++t) sum += scores_[t] / total * v[t * kd + kh * cfg_.head_dim + d];
            attended_[h * cfg_.head_dim + d] = sum;
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
    rms(hidden, normed_, norm_); linear(embedding_, normed_, logits_);
    for (uint32_t i = 0; i < cfg_.vocab; ++i) if (!std::isfinite(logits_[i])) return fail("nonfinite inference logits");
    error_ = nullptr; return true;
}
bool Model::prefill(const uint32_t* ids, uint32_t count) {
    position_ = 0; error_ = nullptr;
    if (!capacity_ || !ids || !count || count > capacity_) return fail("invalid prompt/workspace");
    for (uint32_t t = 0; t < count; ++t) {
        if (ids[t] >= cfg_.vocab) return fail("prompt token outside vocabulary");
        for (uint32_t d = 0; d < cfg_.dim; ++d) hidden_[size_t(t) * cfg_.dim + d] = embedding_.value(size_t(ids[t]) * cfg_.dim + d);
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
    for (uint32_t d = 0; d < cfg_.dim; ++d) hidden_[d] = embedding_.value(size_t(token) * cfg_.dim + d);
    for (uint32_t i = 0; i < cfg_.layers; ++i) {
        keys(layers_[i], i, hidden_, position_);
        if (error_) return false;
        block(layers_[i], i, hidden_, position_, position_ + 1);
        if (error_) return false;
    }
    ++position_; return output(hidden_);
}
uint32_t Model::argmax() const {
    if (!position_ || error_) return cfg_.pad;
    return static_cast<uint32_t>(std::max_element(logits_, logits_ + cfg_.vocab) - logits_);
}
} // namespace ruvi
