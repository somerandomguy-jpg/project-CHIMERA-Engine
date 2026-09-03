#include "benchmark_suite.hpp"
#include "aligned_buffer.hpp"
#include "hugetlb_allocator.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>

static BenchmarkResult evaluate_kernel(const std::string& name,
                                       int src_w, int src_h,
                                       int dst_w, int dst_h,
                                       int iterations,
                                       const std::function<void()>& kernel) {
    for (int i = 0; i < 25; ++i) kernel();

    std::vector<double> timings(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        kernel();
        auto t1 = std::chrono::high_resolution_clock::now();
        timings[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    std::sort(timings.begin(), timings.end());
    double sum = 0.0;
    for (double t : timings) sum += t;

    BenchmarkResult res;
    res.name = name; res.src_w = src_w; res.src_h = src_h; res.dst_w = dst_w; res.dst_h = dst_h; res.iterations = iterations;
    res.min_us = timings.front(); res.max_us = timings.back(); res.mean_us = sum / iterations;
    res.p50_us = timings[iterations / 2];
    res.p95_us = timings[static_cast<size_t>(iterations * 0.95)];
    res.p99_us = timings[static_cast<size_t>(iterations * 0.99)];
    res.fps = 1000000.0 / res.mean_us;
    res.throughput_gp_s = (static_cast<double>(dst_w) * dst_h * res.fps) / 1.0e9;
    return res;
}

std::vector<BenchmarkResult> MicroBenchmarkSuite::run_all(int native_w, int native_h, int render_w, int render_h, int iterations, CorePinnedThreadPool& pool) {
    std::vector<BenchmarkResult> results;
    HugeTlbBuffer<uint32_t, 64> src(render_w * render_h, 0xFF55AAFF);
    HugeTlbBuffer<uint32_t, 64> dst(native_w * native_h, 0);

    // 1. Flagship V4 Quad-Stream
    results.push_back(evaluate_kernel("Tiered Hyper-Omni V4 (Quad-Stream 64px)", render_w, render_h, native_w, native_h, iterations, [&]() { 
        Avx512Upscaler::upscale_tiered_hyper_omni_v4(src.data(), dst.data(), render_w, render_h, native_w, native_h, 0, pool); 
    }));

    // 2. Discrete Proven Paradigms
    results.push_back(evaluate_kernel("Physarum Slime Mold Flux", render_w, render_h, native_w, native_h, iterations, [&]() { Avx512Upscaler::upscale_physarum_slime_mold_2x(src.data(), dst.data(), render_w, render_h, native_w, native_h, pool); }));
    results.push_back(evaluate_kernel("Quantum Ising Spin Ground State", render_w, render_h, native_w, native_h, iterations, [&]() { Avx512Upscaler::upscale_quantum_ising_glass_2x(src.data(), dst.data(), render_w, render_h, native_w, native_h, pool); }));
    results.push_back(evaluate_kernel("Neuromorphic LIF Spiking Mesh", render_w, render_h, native_w, native_h, iterations, [&]() { Avx512Upscaler::upscale_neuromorphic_lif_2x(src.data(), dst.data(), render_w, render_h, native_w, native_h, pool); }));
    results.push_back(evaluate_kernel("GFNI Galois Field Recon", render_w, render_h, native_w, native_h, iterations, [&]() { Avx512Upscaler::upscale_gfni_galois_recon_2x(src.data(), dst.data(), render_w, render_h, native_w, native_h, pool); }));

    return results;
}

void MicroBenchmarkSuite::print_report(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n================================================ [MICRO-BENCHMARK REPORT] ================================================\n"
              << std::left << std::setw(46) << "Tiered Architecture Pipeline"
              << std::right << std::setw(11) << "Mean (ms)"
              << std::setw(11) << "Min (ms)"
              << std::setw(11) << "P50 (ms)"
              << std::setw(11) << "P99 (ms)"
              << std::setw(12) << "FPS"
              << std::setw(14) << "GigaPix/s"
              << "\n--------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(46) << r.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(11) << (r.mean_us / 1000.0)
                  << std::setw(11) << (r.min_us / 1000.0)
                  << std::setw(11) << (r.p50_us / 1000.0)
                  << std::setw(11) << (r.p99_us / 1000.0)
                  << std::setprecision(1) << std::setw(12) << r.fps
                  << std::setprecision(2) << std::setw(14) << r.throughput_gp_s
                  << "\n";
    }
    std::cout << "==========================================================================================================================\n" << std::endl;
}

void MicroBenchmarkSuite::export_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return;
    ofs << "Kernel,SrcWidth,SrcHeight,DstWidth,DstHeight,Iterations,Mean_us,Min_us,Max_us,P50_us,P95_ms,P99_ms,FPS,Throughput_GPix_s\n";
    for (const auto& r : results) {
        ofs << "\"" << r.name << "\"," << r.src_w << "," << r.src_h << "," << r.dst_w << "," << r.dst_h << "," << r.iterations << ","
            << (r.mean_us / 1000.0) << "," << (r.min_us / 1000.0) << "," << (r.max_us / 1000.0) << "," << (r.p50_us / 1000.0) << ","
            << (r.p95_us / 1000.0) << "," << (r.p99_us / 1000.0) << "," << r.fps << "," << r.throughput_gp_s << "\n";
    }
}
