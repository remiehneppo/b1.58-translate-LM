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

    // Test 3: Sample [547, 1332, 7, 1624, 71, 7, 2]
    const uint32_t sample2[] = {547, 1332, 7, 1624, 71, 7, 2};
    written = ruvi::Tokenizer::decode(sample2, 7, buf, sizeof(buf));
    std::printf("Decoded sample 2: %s (len=%zu)\n", buf, written);
    assert(std::strcmp(buf, "Chào! Ca!") == 0);

    // Test 4: Encode Vietnamese text: "Xin chào, bạn khỏe không?"
    // Expected IDs: [2441, 3938, 18, 421, 2507, 337, 37]
    std::string text_vi = "Xin chào, bạn khỏe không?";
    std::vector<uint32_t> ids_vi = ruvi::Tokenizer::encode(text_vi);
    std::printf("Encoded Vietnamese: ");
    for (size_t i = 0; i < ids_vi.size(); ++i) std::printf("%u ", ids_vi[i]);
    std::printf("\n");
    const uint32_t expected_vi[] = {2441, 3938, 18, 421, 2507, 337, 37};
    assert(ids_vi.size() == 7);
    for (size_t i = 0; i < 7; ++i) assert(ids_vi[i] == expected_vi[i]);

    // Test 5: Build prompt with <2ru> (4) and <sep> (3)
    std::vector<uint32_t> prompt_vi2ru = ruvi::Tokenizer::build_prompt(text_vi, ruvi::Tokenizer::TAG_RU);
    assert(prompt_vi2ru.size() == 9);
    assert(prompt_vi2ru.front() == 4);
    assert(prompt_vi2ru.back() == 3);

    // Test 6: Encode Russian text: "Привет, как дела?"
    // Expected IDs: [2494, 1327, 18, 642, 4276, 37]
    std::string text_ru = "Привет, как дела?";
    std::vector<uint32_t> ids_ru = ruvi::Tokenizer::encode(text_ru);
    std::printf("Encoded Russian: ");
    for (size_t i = 0; i < ids_ru.size(); ++i) std::printf("%u ", ids_ru[i]);
    std::printf("\n");
    const uint32_t expected_ru[] = {2494, 1327, 18, 642, 4276, 37};
    assert(ids_ru.size() == 6);
    for (size_t i = 0; i < 6; ++i) assert(ids_ru[i] == expected_ru[i]);

    std::printf("PASS: Standalone C++ Tokenizer encode and decode tests\n");
    return 0;
}
