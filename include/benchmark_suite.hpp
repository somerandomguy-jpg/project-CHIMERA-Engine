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

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "thread_pool.hpp"

struct BenchmarkResult {
    std::string name;
    double min_ms{0.0};
    double max_ms{0.0};
    double mean_ms{0.0};
    double p50_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double fps{0.0};
    double throughput_gp_s{0.0};
};

class MicroBenchmarkSuite {
public:
    static std::vector<BenchmarkResult> run_all(
        int native_w, int native_h, int render_w, int render_h, int iterations,
        CorePinnedThreadPool& pool);

    static void print_report(const std::vector<BenchmarkResult>& results);
    static void export_csv(const std::vector<BenchmarkResult>& results, const std::string& path);
};
