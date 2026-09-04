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

#include <functional>
#include <vulkan/vulkan.h>
#include "aligned_buffer.hpp"
#include "system2_coprocessor_suite.hpp"
#include "thread_pool.hpp"
#include "system2_coprocessor_suite.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

struct KernelProfileMetric {
    std::string name;
    double mean_us;
    double min_us;
    double p50_us;
    double p99_us;
    double throughput_mops;
    std::string config_notes;
};

static KernelProfileMetric profile_kernel(
    const std::string& name,
    const std::string& notes,
    double workload_units,
    int iterations,
    const std::function<void()>& fn) 
{
    for (int i = 0; i < 20; ++i) fn();

    std::vector<double> timings(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        timings[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    std::sort(timings.begin(), timings.end());
    double sum = 0.0;
    for (double t : timings) sum += t;

    KernelProfileMetric m{};
    m.name = name;
    m.config_notes = notes;
    m.mean_us = sum / iterations;
    m.min_us = timings.front();
    m.p50_us = timings[iterations / 2];
    m.p99_us = timings[static_cast<size_t>(iterations * 0.99)];
    m.throughput_mops = (workload_units / (m.mean_us / 1000000.0)) / 1.0e6;
    return m;
}

int main() {
    std::cout << "[1;36m"
              << "========================================================================================================
"
              << "     SYSTEM 2 MULTI-THREADED COPROCESSOR SUITE (6-CORE MASTER-AS-WORKER-0 @ 4.20 GHz)           
"
              << "========================================================================================================[0m

";

    CorePinnedThreadPool pool(6); // Master on Core 0, Workers on Cores 1..5
    const int iterations = 1000;
    std::vector<KernelProfileMetric> metrics;

    // --- KERNEL 1: Frustum Culling 16,384 Spheres ---
    const size_t sphere_count = 16384;
    AlignedBuffer<float, 64> cx(sphere_count, 0.0f);
    AlignedBuffer<float, 64> cy(sphere_count, 0.0f);
    AlignedBuffer<float, 64> cz(sphere_count, 25.0f);
    AlignedBuffer<float, 64> cr(sphere_count, 2.0f);
    BoundingSpheresSoA spheres{cx.data(), cy.data(), cz.data(), cr.data()};

    float planes[6][4] = {
        {1.0f, 0.0f, 0.0f, 100.0f}, {-1.0f, 0.0f, 0.0f, 100.0f},
        {0.0f, 1.0f, 0.0f, 100.0f}, {0.0f, -1.0f, 0.0f, 100.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},   {0.0f, 0.0f, -1.0f, 500.0f}
    };
    AlignedBuffer<uint32_t, 64> idx_counts(sphere_count, 384);
    AlignedBuffer<uint32_t, 64> first_indices(sphere_count, 0);
    AlignedBuffer<int32_t, 64>  vertex_offsets(sphere_count, 0);
    AlignedBuffer<uint8_t, 64>  dst_indirect_bar(sphere_count * sizeof(VkDrawIndexedIndirectCommand), 0);
    uint32_t total_draws_out = 0;

    metrics.push_back(profile_kernel("K1: 320B Frustum Indirect Serializer", "Port 0/1/5 (16k Spheres)", sphere_count, iterations, [&]() {
        System2CoprocessorSuite::FrustumCullAndSerializeIndirect_AVX512(
            spheres, sphere_count, planes, idx_counts.data(), first_indices.data(), vertex_offsets.data(),
            dst_indirect_bar.data(), &total_draws_out
        );
    }));

    // --- KERNEL 2: CRC32 Draw Coalescer (10k Draws) ---
    const size_t draw_count = 10000;
    AlignedBuffer<DrawCallDescriptor, 64> draws(draw_count);
    alignas(64) uint8_t mock_payload[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    for (size_t i = 0; i < draw_count; ++i) {
        draws[i].pso_id = static_cast<uint32_t>(i / 100);
        draws[i].push_constant_size = 16;
        draws[i].push_constant_payload = mock_payload;
    }
    AlignedBuffer<uint32_t, 64> batch_offsets(draw_count, 0);
    AlignedBuffer<uint32_t, 64> batch_counts(draw_count, 0);
    uint32_t total_batches = 0;

    metrics.push_back(profile_kernel("K2: Hardware-CRC32 Batch Sequencer", "Port 1 (10k Draws)", draw_count, iterations, [&]() {
        System2CoprocessorSuite::CoalesceDrawCommands_AVX512(
            draws.data(), draw_count, batch_offsets.data(), batch_counts.data(), &total_batches
        );
    }));

    // --- KERNEL 3: 3D Acoustic FDTD (49,152 Nodes in L2) ---
    const size_t fdtd_nodes = 32 * 32 * 48;
    AlignedBuffer<float, 64> p_next(fdtd_nodes, 0.0f);
    AlignedBuffer<float, 64> p_curr(fdtd_nodes, 1.0f);
    AlignedBuffer<float, 64> p_prev(fdtd_nodes, 0.5f);
    AlignedBuffer<uint8_t, 64> abs_mask(fdtd_nodes, 0);

    metrics.push_back(profile_kernel("K3: 3D Acoustic FDTD Propagator", "100% L2 Cache Resident (192 KB)", fdtd_nodes, iterations, [&]() {
        System2CoprocessorSuite::AcousticSubdomainStep_AVX512(
            p_next.data(), p_curr.data(), p_prev.data(), abs_mask.data(), 343.0f, 0.0001f, 0.05f
        );
    }));

    // --- KERNEL 4: SO(3) Lie Camera Injector ---
    alignas(64) float base_vp[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    alignas(64) CameraMatrixPayload target_slot;

    metrics.push_back(profile_kernel("K4: SO(3) Lie Camera Injector", "Taylor Series 4th-Order", 1.0, iterations, [&]() {
        System2CoprocessorSuite::InjectCameraMatrix_AVX512(
            base_vp, 12.5f, -8.3f, 0.00833f, 0.0022f, &target_slot
        );
    }));

    // --- KERNEL 5 SETUP: Corrected Distinct Vertex Pairs (24,000 Edges across 48,000 Vertices) ---
    const size_t v_count = 48000;
    const size_t e_count = 24000;
    AlignedBuffer<float, 64> px(v_count, 1.0f);
    AlignedBuffer<float, 64> py(v_count, 2.0f);
    AlignedBuffer<float, 64> pz(v_count, 3.0f);
    AlignedBuffer<float, 64> inv_m(v_count, 1.0f);
    MeshParticlesSoA mesh{px.data(), py.data(), pz.data(), inv_m.data(), v_count};

    std::vector<AlignedBuffer<uint32_t, 64>> p1_groups(4), p2_groups(4);
    std::vector<AlignedBuffer<float, 64>> rest_groups(4);
    std::vector<ChromaticEdgeSet> chromatic_sets(4);

    for (int g = 0; g < 4; ++g) {
        p1_groups[g].resize(6000);
        p2_groups[g].resize(6000);
        rest_groups[g].resize(6000, 1.0f);
        
        // Correct initialization: distinct non-overlapping vertex pairs per edge
        for (size_t k = 0; k < 6000; ++k) {
            p1_groups[g][k] = static_cast<uint32_t>((g * 12000 + k * 2) % v_count);
            p2_groups[g][k] = static_cast<uint32_t>((g * 12000 + k * 2 + 1) % v_count);
        }
        chromatic_sets[g] = {p1_groups[g].data(), p2_groups[g].data(), rest_groups[g].data(), 6000};
    }

    AlignedBuffer<uint32_t, 64> single_p1(e_count), single_p2(e_count);
    AlignedBuffer<float, 64> single_rest(e_count, 1.0f);
    for (size_t k = 0; k < e_count; ++k) {
        single_p1[k] = static_cast<uint32_t>((k * 2) % v_count);
        single_p2[k] = static_cast<uint32_t>((k * 2 + 1) % v_count);
    }
    ChromaticEdgeSet single_edge_set{single_p1.data(), single_p2.data(), single_rest.data(), e_count};

    metrics.push_back(profile_kernel("K5: Chromatic PBD Cloth (Single-Core)", "1 Thread Baseline (24k Edges)", e_count, iterations, [&]() {
        System2CoprocessorSuite::RelaxChromaticMesh_AVX512(mesh, single_edge_set);
    }));

    metrics.push_back(profile_kernel("K5: Chromatic PBD Cloth [6-CORE MT]", "6 Physical Cores (24k Edges)", e_count, iterations, [&]() {
        System2CoprocessorSuite::RelaxChromaticMesh_AVX512_MT(mesh, chromatic_sets, pool);
    }));

    // --- KERNEL 6 SETUP: Realistic Sparse Game UI Stencil (10-15% HUD Coverage, 100% L3 Cache Resident) ---
    const int ui_w = 1920, ui_h = 1080;
    const size_t ui_pixels = ui_w * ui_h;
    AlignedBuffer<uint8_t, 64> st_full(ui_pixels, 0); // 0 = Transparent Game Scene
    
    // Draw synthetic UI: Top Compass + Bottom Health/Ammo HUD + Dialogue (~12% Coverage)
    for (int y = 30; y < 80; ++y) std::fill_n(&st_full[y * ui_w + 400], 1120, 255);
    for (int y = 920; y < 1020; ++y) std::fill_n(&st_full[y * ui_w + 200], 1520, 255);
    for (int y = 400; y < 650; ++y) std::fill_n(&st_full[y * ui_w + 100], 450, 255);

    AlignedBuffer<uint8_t, 64> sub_coords(ui_pixels * 4, 32);
    AlignedBuffer<uint8_t, 64> dst_ui(ui_pixels * 4, 0);

    metrics.push_back(profile_kernel("K6: Signed VNNI UI (Single-Core)", "1 Thread Baseline (1080p)", ui_pixels, iterations, [&]() {
        System2CoprocessorSuite::MorphologicalSDFUIComposition_AVX512(
            st_full.data(), st_full.data(), st_full.data(), sub_coords.data(), dst_ui.data(), ui_pixels
        );
    }));

    metrics.push_back(profile_kernel("K6: Signed VNNI UI [6-CORE MT]", "6 Cores + Sparse Skip (1080p)", ui_pixels, iterations, [&]() {
        System2CoprocessorSuite::MorphologicalSDFUIComposition_AVX512_MT(
            st_full.data(), sub_coords.data(), dst_ui.data(), ui_w, ui_h, pool
        );
    }));

    // --- OUTPUT PROFILE MATRIX ---
    std::cout << "[1;32m"
              << "=============================================== [6-CORE MULTI-THREADED PROFILE MATRIX] ===========================================
"
              << "[0m"
              << std::left << std::setw(42) << "Physical Kernel Architecture"
              << std::right << std::setw(11) << "Mean (us)"
              << std::setw(11) << "Min (us)"
              << std::setw(11) << "P50 (us)"
              << std::setw(11) << "P99 (us)"
              << std::setw(16) << "Throughput"
              << std::left << std::setw(30) << "   Hardware Config"
              << "
----------------------------------------------------------------------------------------------------------------------------------
";

    double total_mt_suite_us = 0.0;

    for (const auto& m : metrics) {
        std::cout << std::left << std::setw(42) << m.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(11) << m.mean_us
                  << std::setw(11) << m.min_us
                  << std::setw(11) << m.p50_us
                  << std::setw(11) << m.p99_us
                  << std::setprecision(1) << std::setw(11) << m.throughput_mops << " M/s"
                  << std::left << std::setw(30) << ("   " + m.config_notes)
                  << "
";

        if (m.name.find("MT") != std::string::npos || m.name.find("K1") != std::string::npos || 
            m.name.find("K2") != std::string::npos || m.name.find("K3") != std::string::npos || 
            m.name.find("K4") != std::string::npos) {
            total_mt_suite_us += m.mean_us;
        }
    }

    std::cout << "[1;32m"
              << "==================================================================================================================================
"
              << "[0m";

    double slack_60hz = (16666.6 - total_mt_suite_us) / 1000.0;
    double slack_144hz = (6944.4 - total_mt_suite_us) / 1000.0;
    double duty_60hz = (total_mt_suite_us / 16666.6) * 100.0;
    double duty_144hz = (total_mt_suite_us / 6944.4) * 100.0;

    std::cout << "
[1;33m[HOLISTIC SYSTEM 2 PIPELINE SUMMARY][0m
"
              << "  • Total Combined 6-Core Coprocessor Pass : [1;32m" << std::fixed << std::setprecision(2) << total_mt_suite_us << " us (" << (total_mt_suite_us / 1000.0) << " ms)[0m
"
              << "  • 60Hz  Frame Budget Duty Cycle          : " << std::setprecision(1) << duty_60hz << "% (Slack: " << std::setprecision(2) << slack_60hz << " ms FREE)
"
              << "  • 144Hz Frame Budget Duty Cycle          : " << std::setprecision(1) << duty_144hz << "% (Slack: " << std::setprecision(2) << slack_144hz << " ms FREE)

";

    return 0;
}
