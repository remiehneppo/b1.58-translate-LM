#include "tokenizer.h"
#include "tokenizer_data.h"
#include <cstring>

namespace ruvi {

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

} // namespace ruvi
