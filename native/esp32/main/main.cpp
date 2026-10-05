#include "ruvi.h"
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
    const void* image = nullptr; esp_partition_mmap_handle_t handle;
    esp_err_t status = esp_partition_mmap(partition, 0, size, ESP_PARTITION_MMAP_DATA, &image, &handle);
    if (status != ESP_OK) { ESP_LOGE(tag, "flash mmap failed: %s", esp_err_to_name(status)); return; }
    ruvi::Model model;
    if (!model.load(image, size)) { ESP_LOGE(tag, "%s", model.error()); esp_partition_munmap(handle); return; }
    const int64_t load_us = esp_timer_get_time() - start_load;
    size_t bytes = model.workspace_bytes(CONFIG_RUVI_CONTEXT);
    void* workspace = bytes ? heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT) : nullptr;
    if (!workspace || !model.bind_workspace(workspace, bytes, CONFIG_RUVI_CONTEXT)) {
        ESP_LOGE(tag, "insufficient PSRAM/context; requested %zu bytes", bytes);
        heap_caps_free(workspace); esp_partition_munmap(handle); return;
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
        heap_caps_free(workspace); esp_partition_munmap(handle); return;
    }
    ESP_LOGI(tag, "model=%zu bytes mapped from flash; workspace=%zu bytes in PSRAM", size, bytes);
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
        ESP_LOGI(tag, "token[%zu]=%" PRIu32, generated.size() - 1, token);
        if (token == model.config().eos || generated.size() == CONFIG_RUVI_MAX_NEW) break;
        vTaskDelay(1);
        int64_t step_start = esp_timer_get_time(); success = model.decode(token);
        decode_us += esp_timer_get_time() - step_start; ++steps;
    }
    if (!success) ESP_LOGE(tag, "inference failed: %s", model.error());
    else {
        std::printf("{\"backend\":\"esp32s3-scalar-cpu\",\"model_bytes\":%zu,\"workspace_bytes\":%zu,"
                    "\"image_crc32\":%" PRIu32 ",\"load_ms\":%.3f,"
                    "\"prefill_ms\":%.3f,\"decode_ms\":%.3f,\"decode_steps\":%" PRIu32 ","
                    "\"minimum_free_psram_bytes\":%zu,\"generated_ids\":[",
                    size, bytes, model.image_crc32(), load_us / 1000.0, prefill_us / 1000.0, decode_us / 1000.0, steps,
                    heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
        for (size_t i = 0; i < generated.size(); ++i) { if (i) std::printf(","); std::printf("%" PRIu32, generated[i]); }
        std::printf("]}\n");
    }
    heap_caps_free(workspace); esp_partition_munmap(handle);
}
