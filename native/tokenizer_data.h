#pragma once
#include <cstddef>
#include <cstdint>

namespace ruvi {
namespace tokenizer_data {

constexpr uint32_t VOCAB_SIZE = 8192;
extern const uint32_t TOKEN_OFFSETS[VOCAB_SIZE + 1];
extern const char STRING_POOL[];
extern const uint8_t SUBVOCAB_RU_MASK[VOCAB_SIZE / 8];
extern const uint8_t SUBVOCAB_VI_MASK[VOCAB_SIZE / 8];

} // namespace tokenizer_data
} // namespace ruvi
