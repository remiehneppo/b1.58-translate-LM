#pragma once
#include <cstddef>
#include <cstdint>

namespace ruvi {
namespace tokenizer_data {

constexpr uint32_t VOCAB_SIZE = 8192;
constexpr uint32_t NUM_MERGES = 7929;

struct MergeEntry {
    uint32_t pair_key; // (id1 << 16) | id2
    uint16_t rank;
    uint16_t new_id;
};

extern const uint32_t TOKEN_OFFSETS[VOCAB_SIZE + 1];
extern const char STRING_POOL[];
extern const uint8_t SUBVOCAB_RU_MASK[VOCAB_SIZE / 8];
extern const uint8_t SUBVOCAB_VI_MASK[VOCAB_SIZE / 8];
extern const uint16_t BYTE_TO_TOKEN[256];
extern const MergeEntry MERGE_ENTRIES[NUM_MERGES];

} // namespace tokenizer_data
} // namespace ruvi
