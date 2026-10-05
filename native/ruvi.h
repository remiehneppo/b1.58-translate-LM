#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ruvi {
struct Config {
    uint32_t vocab, dim, layers, heads, kv_heads, head_dim, ff, max_context;
    uint32_t eos, pad, tag_ru, tag_vi;
    float theta, eps;
};
struct Tensor {
    const uint8_t* data = nullptr;
    uint32_t rows = 0, cols = 0, kind = 0;
    float scale = 1;
    float value(size_t index) const;
};
struct Layer {
    Tensor attn_norm, q_norm, k_norm, q, k, v, o, ffn_norm, fc1, fc2;
};
// Weights reference immutable caller-owned image memory (mmap or flash).
// All inference buffers reference caller-owned workspace; no decode allocations.
class Model {
public:
    bool load(const void* image, size_t size);
    size_t workspace_bytes(uint32_t context) const;
    bool bind_workspace(void* memory, size_t bytes, uint32_t context);
    bool prefill(const uint32_t* ids, uint32_t count);
    bool decode(uint32_t token);
    uint32_t argmax() const;
    const float* logits() const { return logits_; }
    const Config& config() const { return cfg_; }
    const char* error() const { return error_; }
    size_t image_bytes() const { return image_bytes_; }
    uint32_t image_crc32() const { return image_crc32_; }
private:
    Config cfg_{};
    Tensor embedding_, norm_;
    std::vector<Layer> layers_;
    size_t image_bytes_ = 0;
    uint32_t image_crc32_ = 0;
    uint32_t capacity_ = 0, position_ = 0;
    const char* error_ = "model not loaded";
    float *cache_ = nullptr, *hidden_ = nullptr, *normed_ = nullptr, *q_ = nullptr;
    float *attended_ = nullptr, *ff_ = nullptr, *temp_ = nullptr, *scores_ = nullptr, *logits_ = nullptr;
    int8_t* quantized_ = nullptr;
    bool fail(const char* message);
    void linear(const Tensor& weight, const float* input, float* output);
    void rms(const float* input, float* output, const Tensor& weight);
    void rope(float* values, uint32_t heads, uint32_t position);
    void keys(const Layer& layer, uint32_t index, const float* input, uint32_t position);
    void block(const Layer& layer, uint32_t index, float* hidden, uint32_t position, uint32_t attend_count);
    bool output(const float* hidden);
};
} // namespace ruvi
