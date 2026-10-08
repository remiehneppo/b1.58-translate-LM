"""Export BPE vocabulary string table and sub-vocabulary bitmasks to C++ source."""
import json
from pathlib import Path
from tokenizers import Tokenizer
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_JSON = ROOT / "ru-vi-tatoeba/tokenizer.json"
OUT_H = ROOT / "native/tokenizer_data.h"
OUT_CPP = ROOT / "native/tokenizer_data.cpp"


def has_cyrillic(text):
    return any("CYRILLIC" in unicodedata.name(c, "") for c in text)


def has_latin(text):
    return any("LATIN" in unicodedata.name(c, "") for c in text)


def export():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    vocab_size = 8192

    pool = bytearray()
    offsets = [0]
    mask_ru = bytearray(vocab_size // 8)
    mask_vi = bytearray(vocab_size // 8)

    for i in range(vocab_size):
        if i == 0:
            s = "<pad>"
        elif i == 1:
            s = "<bos>"
        elif i == 2:
            s = "<eos>"
        elif i == 3:
            s = "<sep>"
        elif i == 4:
            s = "<2ru>"
        elif i == 5:
            s = "<2vi>"
        else:
            s = tokenizer.decode([i])

        encoded = s.encode("utf-8")
        pool.extend(encoded)
        offsets.append(len(pool))

        # Sub-vocab masks
        if i in (0, 1, 2, 3, 4, 5, 6):
            mask_ru[i // 8] |= (1 << (i % 8))
            mask_vi[i // 8] |= (1 << (i % 8))
        else:
            cyr = has_cyrillic(s)
            lat = has_latin(s)
            if cyr and not lat:
                mask_ru[i // 8] |= (1 << (i % 8))
            elif lat and not cyr:
                mask_vi[i // 8] |= (1 << (i % 8))
            else:
                mask_ru[i // 8] |= (1 << (i % 8))
                mask_vi[i // 8] |= (1 << (i % 8))

    print(f"Generated string pool: {len(pool)} bytes, {len(offsets)} offsets.")

    # Write tokenizer_data.h
    OUT_H.write_text(
        """#pragma once
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
""",
        encoding="utf-8",
    )

    # Write tokenizer_data.cpp
    with open(OUT_CPP, "w", encoding="utf-8") as f:
        f.write('#include "tokenizer_data.h"\n\n')
        f.write("namespace ruvi {\nnamespace tokenizer_data {\n\n")

        # TOKEN_OFFSETS
        f.write("const uint32_t TOKEN_OFFSETS[VOCAB_SIZE + 1] = {\n")
        for i in range(0, len(offsets), 8):
            chunk = offsets[i : i + 8]
            f.write("    " + ", ".join(map(str, chunk)) + ",\n")
        f.write("};\n\n")

        # STRING_POOL
        f.write("const char STRING_POOL[] = {\n")
        for i in range(0, len(pool), 16):
            chunk = pool[i : i + 16]
            hex_str = ", ".join(f"static_cast<char>(0x{b:02x})" for b in chunk)
            f.write(f"    {hex_str},\n")
        f.write("    0\n};\n\n")

        # SUBVOCAB_RU_MASK
        f.write("const uint8_t SUBVOCAB_RU_MASK[VOCAB_SIZE / 8] = {\n")
        for i in range(0, len(mask_ru), 16):
            chunk = mask_ru[i : i + 16]
            hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_str},\n")
        f.write("};\n\n")

        # SUBVOCAB_VI_MASK
        f.write("const uint8_t SUBVOCAB_VI_MASK[VOCAB_SIZE / 8] = {\n")
        for i in range(0, len(mask_vi), 16):
            chunk = mask_vi[i : i + 16]
            hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_str},\n")
        f.write("};\n\n")

        f.write("} // namespace tokenizer_data\n} // namespace ruvi\n")

    print(f"Written: {OUT_H} and {OUT_CPP}")


if __name__ == "__main__":
    export()
