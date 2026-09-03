#pragma once
#include <vector>
#include <string>
#include <functional>
#include "avx512_kernel.hpp"
#include "yuv_pipeline.hpp"
#include "thread_pool.hpp"

struct BenchmarkResult {
    std::string name;
    int src_w{0};
    int src_h{0};
    int dst_w{0};
    int dst_h{0};
    int iterations{0};
    double min_us{0.0};
    double max_us{0.0};
    double mean_us{0.0};
    double p50_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double fps{0.0};
    double throughput_gp_s{0.0};
};

class MicroBenchmarkSuite {
public:
    static std::vector<BenchmarkResult> run_all(int native_w, int native_h, int render_w, int render_h, int iterations, CorePinnedThreadPool& pool);
    static void print_report(const std::vector<BenchmarkResult>& results);
    static void export_csv(const std::vector<BenchmarkResult>& results, const std::string& filename);
};
