#include "ruvi.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
uint32_t number(const std::string& text) {
    if (text.empty() || text[0] == '-') throw std::invalid_argument("invalid unsigned integer");
    char* end = nullptr; errno = 0; unsigned long value = std::strtoul(text.c_str(), &end, 10);
    if (errno || *end || value > UINT32_MAX) throw std::invalid_argument("invalid unsigned integer");
    return static_cast<uint32_t>(value);
}
struct Image {
    void* data = MAP_FAILED;
    size_t size = 0;
    explicit Image(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open model image");
        struct stat info{};
        if (fstat(fd, &info) || info.st_size < 80 || uint64_t(info.st_size) > UINT32_MAX) {
            close(fd); throw std::runtime_error("invalid model file size");
        }
        size = static_cast<size_t>(info.st_size);
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
        if (data == MAP_FAILED) throw std::runtime_error("mmap failed");
    }
    ~Image() { if (data != MAP_FAILED) munmap(data, size); }
};
double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(std::ceil(p * values.size())) - 1;
    return values[index];
}
}
int main(int argc, char** argv) {
    try {
        std::string path, prompt, dump;
        uint32_t context = 128, max_new = 8, repeats = 1;
        bool describe = false;
        ruvi::KernelOptions options;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--describe") { describe = true; continue; }
            if (i + 1 == argc) throw std::invalid_argument("missing option value");
            const std::string value = argv[++i];
            if (flag == "--model") path = value;
            else if (flag == "--prompt") prompt = value;
            else if (flag == "--context") context = number(value);
            else if (flag == "--max-new") max_new = number(value);
            else if (flag == "--repeat") repeats = number(value);
            else if (flag == "--logits") dump = value;
            else if (flag == "--backend") {
                if (value == "scalar") options.backend = ruvi::Backend::Scalar;
                else if (value == "tiled") options.backend = ruvi::Backend::Tiled;
                else if (value == "s3") options.backend = ruvi::Backend::S3;
                else throw std::invalid_argument("unknown backend (expected scalar, tiled, or s3)");
            } else if (flag == "--cores") options.cores = number(value);
            else if (flag == "--tile-rows") options.tile_rows = number(value);
            else if (flag == "--microbatch") options.microbatch = number(value);
            else throw std::invalid_argument("unknown option");
        }
        if (path.empty() || !repeats || repeats > 10000) throw std::invalid_argument("--model and valid --repeat required");
        const auto start_load = Clock::now();
        Image image(path); ruvi::Model model;
        model.set_options(options);
        if (!model.load(image.data, image.size)) throw std::runtime_error(model.error());
        const double load_ms = milliseconds(start_load);
        size_t bytes = model.workspace_bytes(context);
        if (!bytes) throw std::invalid_argument("invalid --context");
        if (describe) {
            const auto& cfg = model.config();
            std::cout << "{\"model_bytes\":" << model.image_bytes() << ",\"image_crc32\":" << model.image_crc32() << ",\"workspace_bytes\":" << bytes
                      << ",\"context\":" << context << ",\"vocab\":" << cfg.vocab
                      << ",\"d_model\":" << cfg.dim << ",\"layers\":" << cfg.layers << "}\n";
            return 0;
        }
        std::vector<uint32_t> ids;
        size_t start = 0;
        while (start < prompt.size()) {
            size_t end = prompt.find(',', start);
            ids.push_back(number(prompt.substr(start, end - start)));
            if (end == std::string::npos) break;
            start = end + 1;
            if (start == prompt.size()) throw std::invalid_argument("empty prompt token");
        }
        if (ids.empty() || uint64_t(ids.size()) + max_new > context) throw std::invalid_argument("prompt plus --max-new must fit --context");
        std::vector<float> workspace((bytes + 3) / 4);
        if (!model.bind_workspace(workspace.data(), bytes, context)) throw std::runtime_error(model.error());
        std::vector<double> prefills, decodes, first_tokens, totals;
        std::vector<uint32_t> generated;
        std::vector<float> traces;
        uint64_t decode_steps = 0;
        for (uint32_t run = 0; run < repeats; ++run) {
            generated.clear(); traces.clear();
            double decode_ms = 0;
            const auto start_run = Clock::now();
            if (max_new) {
                const auto start_prefill = Clock::now();
                if (!model.prefill(ids.data(), static_cast<uint32_t>(ids.size()))) throw std::runtime_error(model.error());
                prefills.push_back(milliseconds(start_prefill));
                for (uint32_t step = 0; step < max_new; ++step) {
                    uint32_t token = model.argmax();
                    if (step == 0) first_tokens.push_back(milliseconds(start_run));
                    generated.push_back(token);
                    // Traces are diagnostic host-only allocations, outside decode timing.
                    if (!dump.empty()) traces.insert(traces.end(), model.logits(), model.logits() + model.config().vocab);
                    if (token == model.config().eos || step + 1 == max_new) break;
                    const auto start_decode = Clock::now();
                    if (!model.decode(token)) throw std::runtime_error(model.error());
                    decode_ms += milliseconds(start_decode); ++decode_steps;
                }
            } else { prefills.push_back(0); first_tokens.push_back(0); }
            decodes.push_back(decode_ms); totals.push_back(milliseconds(start_run));
        }
        if (!dump.empty()) {
            std::ofstream file(dump, std::ios::binary);
            file.write(reinterpret_cast<const char*>(traces.data()), static_cast<std::streamsize>(traces.size() * sizeof(float)));
            if (!file) throw std::runtime_error("cannot write logits trace");
        }
        struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        long peak_rss_kib = usage.ru_maxrss;
        const char* rss_source = "getrusage";
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.compare(0, 6, "VmHWM:") == 0) {
                peak_rss_kib = std::stol(line.substr(6)); rss_source = "proc-self-VmHWM"; break;
            }
        }
        double decode_sum = 0; for (double t : decodes) decode_sum += t;
        const char* backend_str = (model.effective_backend() == ruvi::Backend::Scalar ? "scalar-cpu" :
                                  (model.effective_backend() == ruvi::Backend::S3 ? "esp32s3-simd" : "tiled-cpu"));
        std::cout << "{\"backend\":\"" << backend_str << "\""
                  << ",\"model_bytes\":" << model.image_bytes()
                  << ",\"image_crc32\":" << model.image_crc32() << ",\"load_ms\":" << load_ms
                  << ",\"workspace_bytes\":" << bytes << ",\"peak_rss_kib\":" << peak_rss_kib
                  << ",\"peak_rss_source\":\"" << rss_source << "\""
                  << ",\"context\":" << context << ",\"prompt_tokens\":" << ids.size()
                  << ",\"repeats\":" << repeats << ",\"prefill_first_run_ms\":" << prefills.front()
                  << ",\"prefill_p50_ms\":" << percentile(prefills, 0.5)
                  << ",\"prefill_p95_ms\":" << percentile(prefills, 0.95)
                  << ",\"ttft_p50_ms\":" << percentile(first_tokens, 0.5)
                  << ",\"decode_p50_ms\":" << percentile(decodes, 0.5)
                  << ",\"total_p50_ms\":" << percentile(totals, 0.5)
                  << ",\"decode_steps\":" << decode_steps << ",\"decode_tokens_per_second\":";
        if (decode_sum > 0) std::cout << 1000.0 * decode_steps / decode_sum; else std::cout << "null";
        std::cout << ",\"generated_ids\":[";
        for (size_t i = 0; i < generated.size(); ++i) { if (i) std::cout << ','; std::cout << generated[i]; }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << "ruvi-native: " << error.what() << '\n'; return 1;
    }
}
