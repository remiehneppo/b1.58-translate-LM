#include "parallel.h"

#if defined(ESP_PLATFORM)

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

namespace ruvi {
namespace {

static TaskHandle_t s_worker_handle = nullptr;
static SemaphoreHandle_t s_done_sem = nullptr;
static ParallelTask s_current_task = nullptr;
static void* s_current_context = nullptr;
static uint32_t s_current_parts = 1;
static volatile bool s_is_busy = false;

static void worker_entry(void* /*arg*/) {
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (s_current_task) {
            for (uint32_t i = 1; i < s_current_parts; ++i) {
                s_current_task(s_current_context, i);
            }
        }
        xSemaphoreGive(s_done_sem);
    }
}

} // namespace

void parallel_init() {
    if (s_worker_handle != nullptr) return;
    s_done_sem = xSemaphoreCreateBinary();
    if (!s_done_sem) return;

    BaseType_t res = xTaskCreatePinnedToCore(
        worker_entry,
        "ruvi_worker1",
        4096,
        nullptr,
        configMAX_PRIORITIES - 2,
        &s_worker_handle,
        1
    );
    if (res != pdPASS) {
        s_worker_handle = nullptr;
        vSemaphoreDelete(s_done_sem);
        s_done_sem = nullptr;
    }
}

void parallel_shutdown() {
    if (s_worker_handle) {
        vTaskDelete(s_worker_handle);
        s_worker_handle = nullptr;
    }
    if (s_done_sem) {
        vSemaphoreDelete(s_done_sem);
        s_done_sem = nullptr;
    }
    s_is_busy = false;
}

void parallel_run(ParallelTask task, void* context, uint32_t num_parts) {
    if (num_parts < 2 || !task) {
        if (task && num_parts >= 1) task(context, 0);
        return;
    }

    if (s_is_busy || s_worker_handle == nullptr || xTaskGetCurrentTaskHandle() == s_worker_handle) {
        for (uint32_t i = 0; i < num_parts; ++i) {
            task(context, i);
        }
        return;
    }

    s_is_busy = true;
    s_current_task = task;
    s_current_context = context;
    s_current_parts = num_parts;

    // Wake Core 1
    xTaskNotifyGive(s_worker_handle);

    // Run part 0 on caller (Core 0)
    task(context, 0);

    // Wait for Core 1 to complete
    xSemaphoreTake(s_done_sem, portMAX_DELAY);
    s_is_busy = false;
}

} // namespace ruvi

#else

namespace ruvi {

void parallel_init() {}
void parallel_shutdown() {}

void parallel_run(ParallelTask task, void* context, uint32_t num_parts) {
    if (!task) return;
    for (uint32_t i = 0; i < num_parts; ++i) {
        task(context, i);
    }
}

} // namespace ruvi

#endif
