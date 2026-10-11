#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ruvi {

class Tokenizer {
public:
    static constexpr uint32_t PAD = 0;
    static constexpr uint32_t BOS = 1;
    static constexpr uint32_t EOS = 2;
    static constexpr uint32_t SEP = 3;
    static constexpr uint32_t TAG_RU = 4;
    static constexpr uint32_t TAG_VI = 5;
    static constexpr uint32_t VOCAB_SIZE = 8192;

    // Encodes UTF-8 string into token IDs using exact BPE merges.
    static std::vector<uint32_t> encode(const char* text, size_t len);
    static std::vector<uint32_t> encode(const std::string& text);

    // Builds the full translation prompt IDs: [TAG] + encoded_ids + [SEP]
    // tag is Tokenizer::TAG_RU (4) for vi2ru or Tokenizer::TAG_VI (5) for ru2vi.
    static std::vector<uint32_t> build_prompt(const std::string& text, uint32_t tag);

    // Decodes a single token ID in O(1) time without heap allocations.
    // Returns true if token_id < VOCAB_SIZE.
    static bool decode_token(uint32_t token_id, const char** out_str, size_t* out_len);

    // Decodes an array of token IDs into a null-terminated UTF-8 string buffer.
    // Automatically stops if EOS (token 2) is encountered.
    // Returns number of bytes written (excluding terminating null).
    static size_t decode(const uint32_t* ids, size_t count, char* out_buf, size_t max_buf_len);
    static std::string decode(const std::vector<uint32_t>& ids);
};

} // namespace ruvi
