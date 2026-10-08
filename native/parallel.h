#pragma once
#include <cstdint>

namespace ruvi {

typedef void (*ParallelTask)(void* context, uint32_t part_index);

// Dispatches a task across cores (part 0 on caller core, part 1 on worker core).
// On platforms without multi-core support or if worker is inactive, executes sequentially.
void parallel_init();
void parallel_run(ParallelTask task, void* context, uint32_t num_parts);
void parallel_shutdown();

} // namespace ruvi
