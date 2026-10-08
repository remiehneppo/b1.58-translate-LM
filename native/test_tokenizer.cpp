#include "tokenizer.h"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    // Test 1: Special tokens
    const char* str = nullptr;
    size_t len = 0;
    assert(ruvi::Tokenizer::decode_token(ruvi::Tokenizer::PAD, &str, &len));
    assert(std::strncmp(str, "<pad>", len) == 0);

    // Test 2: Sample Russian translation [2494, 1327, 18, 617, 1535, 1271, 1737, 4363, 37, 2]
    // Expected: "Привет, ты хорошо себя чувствуешь?"
    const uint32_t sample1[] = {2494, 1327, 18, 617, 1535, 1271, 1737, 4363, 37, 2};
    char buf[256];
    size_t written = ruvi::Tokenizer::decode(sample1, 10, buf, sizeof(buf));
    std::printf("Decoded sample 1: %s (len=%zu)\n", buf, written);
    assert(std::strcmp(buf, "Привет, ты хорошо себя чувствуешь?") == 0);

    // Test 3: Sample Russian [547, 1332, 7, 1624, 71, 7, 2]
    // Expected: "Chào! Ca!"
    const uint32_t sample2[] = {547, 1332, 7, 1624, 71, 7, 2};
    written = ruvi::Tokenizer::decode(sample2, 7, buf, sizeof(buf));
    std::printf("Decoded sample 2: %s (len=%zu)\n", buf, written);
    assert(std::strcmp(buf, "Chào! Ca!") == 0);

    std::printf("PASS: Standalone C++ Tokenizer decode tests\n");
    return 0;
}
