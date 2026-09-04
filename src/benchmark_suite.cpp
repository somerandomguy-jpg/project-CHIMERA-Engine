/*
 * Project CHIMERA Engine: AVX-512 Heterogeneous Graphics & Vector Coprocessor
 * Copyright (C) 2026 somerandomguy-jpg <https://github.com/somerandomguy-jpg>
 *
 * This file is part of Project CHIMERA Engine.
 *
 * Project CHIMERA Engine is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Project CHIMERA Engine is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Project CHIMERA Engine. If not, see <https://www.gnu.org/licenses/>.
 */

#include "benchmark_suite.hpp"
#include "avx512_kernel.hpp"
#include "hugetlb_allocator.hpp"
#include "aligned_buffer.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <numeric>

std::vector<BenchmarkResult> MicroBenchmarkSuite::run_all(
    int native_w, int native_h, int render_w, int render_h, int iterations,
    CorePinnedThreadPool& pool)
{
    std::vector<BenchmarkResult> results;
    const size_t src_size = static_cast<size_t>(render_w * render_h);
    const size_t dst_size = static_cast<size_t>(native_w * native_h);

    AlignedBuffer<uint32_t, 64> src(src_size, 0xFF55AAFF);
    AlignedBuffer<uint32_t, 64> dst(dst_size, 0);

    auto benchmark_kernel = [&](const std::string& name, auto&& fn) {
        std::vector<double> latencies_us;
        latencies_us.reserve(iterations);

        // Warmup passes
        for (int i = 0; i < 20; ++i) {
            fn();
        }

        for (int i = 0; i < iterations; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            latencies_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        std::sort(latencies_us.begin(), latencies_us.end());
        double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        double mean = sum / iterations;
        double p50 = latencies_us[iterations * 50 / 100];
        double p95 = latencies_us[iterations * 95 / 100];
        double p99 = latencies_us[iterations * 99 / 100];
        double min_v = latencies_us.front();
        double max_v = latencies_us.back();

        BenchmarkResult r;
        r.name = name;
        r.mean_ms = mean / 1000.0;
        r.min_ms = min_v / 1000.0;
        r.max_ms = max_v / 1000.0;
        r.p50_us = p50;
        r.p95_us = p95;
        r.p99_us = p99;
        r.fps = 1000.0 / r.mean_ms;
        r.throughput_gp_s = (static_cast<double>(dst_size) * r.fps) / 1e9;
        results.push_back(r);
    };

    benchmark_kernel("Tiered Hyper-Omni V4 (Quad-Stream 64px)", [&]() {
        Avx512Upscaler::upscale_tiered_hyper_omni_v4(src.data(), dst.data(), render_w, render_h, native_w, native_h, 0, pool);
    });

    benchmark_kernel("Physarum Slime Mold Flux", [&]() {
        Avx512Upscaler::upscale_physarum_slime_mold_2x(src.data(), dst.data(), render_w, render_h, native_w, native_h, pool);
    });

    benchmark_kernel("Quantum Ising Spin Ground State", [&]() {
        Avx512Upscaler::upscale_quantum_ising_glass_2x(src.data(), dst.data(), render_w, render_h, native_w, native_h, pool);
    });

    benchmark_kernel("Neuromorphic LIF Spiking Mesh", [&]() {
        Avx512Upscaler::upscale_neuromorphic_lif_2x(src.data(), dst.data(), render_w, render_h, native_w, native_h, pool);
    });

    benchmark_kernel("GFNI Galois Field Recon", [&]() {
        Avx512Upscaler::upscale_gfni_galois_recon_2x(src.data(), dst.data(), render_w, render_h, native_w, native_h, pool);
    });

    return results;
}

void MicroBenchmarkSuite::print_report(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n================================================ [MICRO-BENCHMARK REPORT] ================================================\n"
              << std::left << std::setw(46) << "Tiered Architecture Pipeline"
              << std::right
              << std::setw(12) << "Mean (ms)"
              << std::setw(11) << "Min (ms)"
              << std::setw(11) << "P50 (ms)"
              << std::setw(11) << "P99 (ms)"
              << std::setw(13) << "FPS"
              << std::setw(14) << "GigaPix/s\n"
              << "--------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::cout << std::left  << std::setw(46) << r.name
                  << std::right
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << r.mean_ms
                  << std::setw(11) << r.min_ms
                  << std::setw(11) << (r.p50_us / 1000.0)
                  << std::setw(11) << (r.p99_us / 1000.0)
                  << std::setprecision(1)
                  << std::setw(13) << r.fps
                  << std::setprecision(2)
                  << std::setw(14) << r.throughput_gp_s
                  << "\n";
    }
    std::cout << "==========================================================================================================================\n" << std::endl;
}

void MicroBenchmarkSuite::export_csv(const std::vector<BenchmarkResult>& results, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return;
    ofs << "Kernel,SrcWidth,SrcHeight,DstWidth,DstHeight,Iterations,Mean_us,Min_us,Max_us,P50_us,P95_ms,P99_ms,FPS,Throughput_GPix_s\n";
    for (const auto& r : results) {
        ofs << r.name << ","
            << 960 << "," << 540 << "," << 1920 << "," << 1080 << "," << 1000 << ","
            << (r.mean_ms * 1000.0) << "," << (r.min_ms * 1000.0) << "," << (r.max_ms * 1000.0) << ","
            << r.p50_us << "," << (r.p95_us / 1000.0) << "," << (r.p99_us / 1000.0) << ","
            << r.fps << "," << r.throughput_gp_s << "\n";
    }
}
