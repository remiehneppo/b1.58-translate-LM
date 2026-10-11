#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ruvi {
struct Config {
    uint32_t vocab, dim, layers, heads, kv_heads, head_dim, ff, max_context;
    uint32_t eos, pad, tag_ru, tag_vi;
    float theta, eps, attn_softcap, final_softcap;
};
struct Tensor {
    const uint8_t* data = nullptr;
    uint32_t rows = 0, cols = 0, kind = 0;
    float scale = 1;
    float value(size_t index) const;
};
enum class Backend { Scalar, Tiled, S3, RV1103 };
struct KernelOptions {
    Backend backend = Backend::Scalar;
    uint32_t cores = 1, tile_rows = 16, microbatch = 16;
    bool subvocab = true;
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
    void set_options(const KernelOptions& options);
    const KernelOptions& options() const { return options_; }
    Backend effective_backend() const { return effective_backend_; }
    size_t scratch_bytes() const;
    bool bind_scratch(void* scratch, size_t bytes);
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
    KernelOptions options_{};
    Backend effective_backend_ = Backend::Scalar;
    uint32_t target_lang_ = 0;

    void* scratch_mem_ = nullptr;
    size_t scratch_bytes_ = 0;
    int8_t* tile_scratch_[2] = {nullptr, nullptr};
    size_t tile_scratch_bytes_ = 0;
    int8_t* quant_scratch_ = nullptr;
    size_t quant_scratch_bytes_ = 0;

    float *cache_ = nullptr, *hidden_ = nullptr, *normed_ = nullptr, *q_ = nullptr;
    float *attended_ = nullptr, *ff_ = nullptr, *temp_ = nullptr, *scores_ = nullptr, *logits_ = nullptr;
    float *rope_cos_ = nullptr, *rope_sin_ = nullptr;
    int8_t* quantized_ = nullptr;

    bool fail(const char* message);
    void linear(const Tensor& weight, const float* input, float* output);
    void linear_int8(const Tensor& weight, const int8_t* input_quant, float multiplier, float* output, const uint8_t* subvocab_mask = nullptr);
    void linear_ternary(const Tensor& weight, const int8_t* input_quant, float multiplier, float* output, uint32_t tokens = 1, size_t in_stride = 0, size_t out_stride = 0);
    void rms(const float* input, float* output, const Tensor& weight);
    void rms_heads(const float* input, float* output, const Tensor& weight, uint32_t heads, uint32_t head_dim);
    void quantize_row(const float* input, uint32_t cols, int8_t* out, float& multiplier);
    void rope(float* values, uint32_t heads, uint32_t position);
    void keys(const Layer& layer, uint32_t index, const float* input, uint32_t position);
    void block(const Layer& layer, uint32_t index, float* hidden, uint32_t position, uint32_t attend_count);
    bool output(const float* hidden);
    void init_scratch();
};
} // namespace ruvi
