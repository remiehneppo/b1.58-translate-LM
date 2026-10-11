#include "tokenizer.h"
#include "tokenizer_data.h"
#include <algorithm>
#include <cstring>

namespace ruvi {
namespace {

inline bool find_merge(uint16_t id1, uint16_t id2, uint16_t& rank, uint16_t& new_id) {
    const uint32_t key = (static_cast<uint32_t>(id1) << 16) | id2;
    size_t low = 0, high = tokenizer_data::NUM_MERGES;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (tokenizer_data::MERGE_ENTRIES[mid].pair_key < key) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low < tokenizer_data::NUM_MERGES && tokenizer_data::MERGE_ENTRIES[low].pair_key == key) {
        rank = tokenizer_data::MERGE_ENTRIES[low].rank;
        new_id = tokenizer_data::MERGE_ENTRIES[low].new_id;
        return true;
    }
    return false;
}

} // namespace

std::vector<uint32_t> Tokenizer::encode(const char* text, size_t len) {
    if (!text || len == 0) return {};

    // 1. Convert each byte to its base token ID
    std::vector<uint16_t> tokens;
    tokens.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        uint8_t byte_val = static_cast<uint8_t>(text[i]);
        tokens.push_back(tokenizer_data::BYTE_TO_TOKEN[byte_val]);
    }

    // 2. Iteratively merge adjacent pairs with lowest rank
    while (tokens.size() > 1) {
        uint16_t min_rank = 0xFFFF;
        size_t min_idx = SIZE_MAX;
        uint16_t min_new_id = 0;

        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            uint16_t rank = 0, new_id = 0;
            if (find_merge(tokens[i], tokens[i + 1], rank, new_id)) {
                if (rank < min_rank) {
                    min_rank = rank;
                    min_idx = i;
                    min_new_id = new_id;
                }
            }
        }

        if (min_idx == SIZE_MAX) break;

        tokens[min_idx] = min_new_id;
        tokens.erase(tokens.begin() + min_idx + 1);
    }

    return std::vector<uint32_t>(tokens.begin(), tokens.end());
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) {
    return encode(text.data(), text.size());
}

std::vector<uint32_t> Tokenizer::build_prompt(const std::string& text, uint32_t tag) {
    std::vector<uint32_t> prompt;
    prompt.push_back(tag);
    std::vector<uint32_t> body = encode(text);
    prompt.insert(prompt.end(), body.begin(), body.end());
    prompt.push_back(SEP);
    return prompt;
}

bool Tokenizer::decode_token(uint32_t token_id, const char** out_str, size_t* out_len) {
    if (token_id >= tokenizer_data::VOCAB_SIZE) {
        if (out_str) *out_str = "";
        if (out_len) *out_len = 0;
        return false;
    }
    const uint32_t start = tokenizer_data::TOKEN_OFFSETS[token_id];
    const uint32_t end = tokenizer_data::TOKEN_OFFSETS[token_id + 1];
    if (out_str) *out_str = tokenizer_data::STRING_POOL + start;
    if (out_len) *out_len = static_cast<size_t>(end - start);
    return true;
}

size_t Tokenizer::decode(const uint32_t* ids, size_t count, char* out_buf, size_t max_buf_len) {
    if (!out_buf || max_buf_len == 0) return 0;
    out_buf[0] = '\0';
    if (!ids || count == 0) return 0;

    size_t written = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t id = ids[i];
        if (id == EOS) break;
        if (id == PAD || id == BOS || id == SEP || id == TAG_RU || id == TAG_VI) {
            continue; // Skip control tags in user-facing translation text
        }
        const char* piece = nullptr;
        size_t piece_len = 0;
        if (decode_token(id, &piece, &piece_len) && piece_len > 0) {
            if (written + piece_len + 1 > max_buf_len) {
                piece_len = max_buf_len - 1 - written;
            }
            std::memcpy(out_buf + written, piece, piece_len);
            written += piece_len;
            if (written + 1 >= max_buf_len) break;
        }
    }
    out_buf[written] = '\0';
    return written;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) {
    char buf[4096];
    size_t written = decode(ids.data(), ids.size(), buf, sizeof(buf));
    return std::string(buf, written);
}

} // namespace ruvi
