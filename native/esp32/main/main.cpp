#include "ruvi.h"
#include "parallel.h"
#include "tokenizer.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <string>
#include <vector>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

extern "C" void app_main() {
    const char* tag = "ruvi";
    const auto* partition = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!partition) { ESP_LOGE(tag, "model partition missing"); return; }
    uint8_t header[80];
    if (esp_partition_read(partition, 0, header, sizeof(header)) != ESP_OK || std::memcmp(header, "RUVI0001", 8)) {
        ESP_LOGE(tag, "flash a RUVI0001 image into model partition first"); return;
    }
    size_t size = uint32_t(header[64]) | (uint32_t(header[65]) << 8) | (uint32_t(header[66]) << 16) | (uint32_t(header[67]) << 24);
    if (size < sizeof(header) || size > partition->size) { ESP_LOGE(tag, "model exceeds partition"); return; }
    const int64_t start_load = esp_timer_get_time();
    const void* image = nullptr; esp_partition_mmap_handle_t handle = 0;
    esp_err_t status = esp_partition_mmap(partition, 0, size, ESP_PARTITION_MMAP_DATA, &image, &handle);
    if (status != ESP_OK) { ESP_LOGE(tag, "flash mmap failed: %s", esp_err_to_name(status)); return; }

    void* model_psram = nullptr;
#if defined(CONFIG_RUVI_MODEL_IN_PSRAM)
    model_psram = heap_caps_aligned_alloc(16, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (model_psram) {
        std::memcpy(model_psram, image, size);
        esp_partition_munmap(handle);
        handle = 0;
        image = model_psram;
        ESP_LOGI(tag, "Loaded entire model (%zu bytes) into Octal PSRAM", size);
    } else {
        ESP_LOGW(tag, "Failed to allocate %zu bytes in PSRAM for model, falling back to Flash XIP", size);
    }
#endif

    ruvi::Model model;
    if (!model.load(image, size)) {
        ESP_LOGE(tag, "%s", model.error());
        if (model_psram) heap_caps_free(model_psram);
        else if (handle) esp_partition_munmap(handle);
        return;
    }
    const int64_t load_us = esp_timer_get_time() - start_load;

    ruvi::KernelOptions opts;
#if defined(CONFIG_RUVI_BACKEND_S3)
    opts.backend = ruvi::Backend::S3;
#elif defined(CONFIG_RUVI_BACKEND_TILED)
    opts.backend = ruvi::Backend::Tiled;
#else
    opts.backend = ruvi::Backend::Scalar;
#endif
#if defined(CONFIG_RUVI_CORES)
    opts.cores = CONFIG_RUVI_CORES;
#endif
#if defined(CONFIG_RUVI_TILE_ROWS)
    opts.tile_rows = CONFIG_RUVI_TILE_ROWS;
#endif
#if defined(CONFIG_RUVI_MICROBATCH)
    opts.microbatch = CONFIG_RUVI_MICROBATCH;
#endif
    if (opts.cores > 1) {
        ruvi::parallel_init();
    }
    model.set_options(opts);

    size_t bytes = model.workspace_bytes(CONFIG_RUVI_CONTEXT);
    void* workspace = bytes ? heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT) : nullptr;
    if (!workspace || !model.bind_workspace(workspace, bytes, CONFIG_RUVI_CONTEXT)) {
        ESP_LOGE(tag, "insufficient PSRAM/context; requested %zu bytes", bytes);
        heap_caps_free(workspace); esp_partition_munmap(handle); return;
    }

    size_t sram_needed = model.scratch_bytes();
    void* sram_scratch = sram_needed ? heap_caps_aligned_alloc(16, sram_needed, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) : nullptr;
    if (sram_scratch) {
        if (!model.bind_scratch(sram_scratch, sram_needed)) {
            ESP_LOGW(tag, "binding internal SRAM scratch failed, using workspace fallback");
            heap_caps_free(sram_scratch);
            sram_scratch = nullptr;
        } else {
            ESP_LOGI(tag, "bound %zu bytes tile scratch in internal SRAM", sram_needed);
        }
    }

    std::vector<uint32_t> prompt;
    const char* p = CONFIG_RUVI_PROMPT;
    bool valid = *p != '\0';
    while (valid && *p) {
        if (*p < '0' || *p > '9') { valid = false; break; }
        char* end = nullptr; errno = 0; unsigned long token = std::strtoul(p, &end, 10);
        if (errno || token >= model.config().vocab || (*end && *end != ',')) { valid = false; break; }
        prompt.push_back(static_cast<uint32_t>(token));
        if (*end == ',' && !end[1]) { valid = false; break; }
        p = *end ? end + 1 : end;
    }
    if (!valid || prompt.size() + CONFIG_RUVI_MAX_NEW > CONFIG_RUVI_CONTEXT) {
        ESP_LOGE(tag, "invalid prompt or prompt plus max-new exceeds context");
        if (sram_scratch) heap_caps_free(sram_scratch);
        heap_caps_free(workspace); esp_partition_munmap(handle); return;
    }
    ESP_LOGI(tag, "model=%zu bytes mapped from flash; workspace=%zu bytes in PSRAM", size, bytes);
    ESP_LOGI(tag, "backend=%s cores=%" PRIu32 " tile_rows=%" PRIu32,
             (model.effective_backend() == ruvi::Backend::S3) ? "s3-simd" :
             ((model.effective_backend() == ruvi::Backend::Tiled) ? "tiled" : "scalar"),
             opts.cores, opts.tile_rows);
    ESP_LOGI(tag, "architecture: vocab=%" PRIu32 " dim=%" PRIu32 " layers=%" PRIu32,
             model.config().vocab, model.config().dim, model.config().layers);
    vTaskDelay(1);
    const int64_t start = esp_timer_get_time();
    bool success = model.prefill(prompt.data(), static_cast<uint32_t>(prompt.size()));
    const int64_t prefill_us = esp_timer_get_time() - start;
    int64_t decode_us = 0; uint32_t steps = 0;
    std::vector<uint32_t> generated;
    while (success && generated.size() < CONFIG_RUVI_MAX_NEW) {
        uint32_t token = model.argmax(); generated.push_back(token);
        const char* piece = nullptr; size_t piece_len = 0;
        ruvi::Tokenizer::decode_token(token, &piece, &piece_len);
        std::string piece_str(piece, piece_len);
        ESP_LOGI(tag, "token[%zu]=%" PRIu32 " ('%s')", generated.size() - 1, token, piece_str.c_str());
        if (token == model.config().eos || generated.size() == CONFIG_RUVI_MAX_NEW) break;
        vTaskDelay(1);
        int64_t step_start = esp_timer_get_time(); success = model.decode(token);
        decode_us += esp_timer_get_time() - step_start; ++steps;
    }
    if (!success) ESP_LOGE(tag, "inference failed: %s", model.error());
    else {
        char translated_text[256] = {};
        ruvi::Tokenizer::decode(generated.data(), generated.size(), translated_text, sizeof(translated_text));
        ESP_LOGI(tag, "========================================");
        ESP_LOGI(tag, "TRANSLATION OUTPUT: %s", translated_text);
        ESP_LOGI(tag, "========================================");

        const char* backend_name = (model.effective_backend() == ruvi::Backend::S3) ? "esp32s3-simd" :
                                   ((model.effective_backend() == ruvi::Backend::Tiled) ? "esp32s3-tiled-cpu" : "esp32s3-scalar-cpu");
        std::printf("{\"backend\":\"%s\",\"model_bytes\":%zu,\"workspace_bytes\":%zu,\"sram_scratch_bytes\":%zu,"
                    "\"image_crc32\":%" PRIu32 ",\"load_ms\":%.3f,"
                    "\"prefill_ms\":%.3f,\"decode_ms\":%.3f,\"decode_steps\":%" PRIu32 ","
                    "\"minimum_free_psram_bytes\":%zu,\"minimum_free_sram_bytes\":%zu,"
                    "\"translation\":\"%s\",\"generated_ids\":[",
                    backend_name, size, bytes, sram_scratch ? sram_needed : 0, model.image_crc32(),
                    load_us / 1000.0, prefill_us / 1000.0, decode_us / 1000.0, steps,
                    heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM),
                    heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
                    translated_text);
        for (size_t i = 0; i < generated.size(); ++i) { if (i) std::printf(","); std::printf("%" PRIu32, generated[i]); }
        std::printf("]}\n");
    }
    if (opts.cores > 1) {
        ruvi::parallel_shutdown();
    }
    if (sram_scratch) heap_caps_free(sram_scratch);
    heap_caps_free(workspace);
    if (model_psram) {
        heap_caps_free(model_psram);
    } else if (handle) {
        esp_partition_munmap(handle);
    }
}
