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

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <functional>
#include <vulkan/vulkan.h>

#include "system2_coprocessor_suite.hpp"
#include "thread_pool.hpp"
#include "aligned_buffer.hpp"

struct KernelProfileMetric {
    std::string name;
    std::string port_info;
    double throughput_m;
    double latency_us;
    double fps;
};

KernelProfileMetric profile_kernel(
    const std::string& name,
    const std::string& port_info,
    double work_units,
    int iterations,
    const std::function<void()>& fn)
{
    for (int i = 0; i < 20; ++i) fn(); // Warmup

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double avg_us = total_us / iterations;
    double throughput_m = (work_units / (avg_us / 1e6)) / 1e6;
    double fps = 1e6 / avg_us;

    return {name, port_info, throughput_m, avg_us, fps};
}

int main() {
    std::cout << "========================================================================================================\n"
              << "     SYSTEM 2 MULTI-THREADED COPROCESSOR SUITE (6-CORE MASTER-AS-WORKER-0 @ 4.20 GHz)                   \n"
              << "========================================================================================================\n";

    CorePinnedThreadPool pool(6);
    std::vector<KernelProfileMetric> metrics;
    const int iterations = 500;

    // K1: Frustum Culling (16,384 Spheres)
    const size_t sphere_count = 16384;
    AlignedBuffer<float, 64> cx(sphere_count, 0.0f);
    AlignedBuffer<float, 64> cy(sphere_count, 0.0f);
    AlignedBuffer<float, 64> cz(sphere_count, 25.0f);
    AlignedBuffer<float, 64> cr(sphere_count, 2.0f);
    BoundingSpheresSoA spheres{cx.data(), cy.data(), cz.data(), cr.data()};
    float planes[6][4] = {
        { 1, 0, 0, 50}, {-1, 0, 0, 50},
        { 0, 1, 0, 50}, { 0,-1, 0, 50},
        { 0, 0, 1, 10}, { 0, 0,-1, 100}
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

    // K2: Draw Call Coalescer (10,000 Draws)
    const size_t draw_count = 10000;
    AlignedBuffer<DrawCallDescriptor, 64> draws(draw_count);
    for (size_t i = 0; i < draw_count; ++i) {
        draws[i].pso_id = static_cast<uint32_t>(i / 100);
        draws[i].pipeline_id = static_cast<uint32_t>(i / 100);
        draws[i].material_id = static_cast<uint32_t>((i / 10) % 5);
    }
    AlignedBuffer<uint32_t, 64> batch_offsets(draw_count, 0);
    AlignedBuffer<uint32_t, 64> batch_counts(draw_count, 0);
    uint32_t total_batches = 0;

    metrics.push_back(profile_kernel("K2: Hardware-CRC32 Batch Sequencer", "Port 1 (10k Draws)", draw_count, iterations, [&]() {
        System2CoprocessorSuite::CoalesceDrawCommands_AVX512(
            draws.data(), draw_count, batch_offsets.data(), batch_counts.data(), &total_batches
        );
    }));

    // K3: Acoustic FDTD Propagator (49,152 Nodes)
    const size_t fdtd_nodes = 49152;
    AlignedBuffer<float, 64> p_next(fdtd_nodes, 0.0f);
    AlignedBuffer<float, 64> p_curr(fdtd_nodes, 1.0f);
    AlignedBuffer<float, 64> p_prev(fdtd_nodes, 0.5f);
    AlignedBuffer<uint8_t, 64> abs_mask(fdtd_nodes, 0);

    metrics.push_back(profile_kernel("K3: 3D Acoustic FDTD Propagator", "100% L2 Cache Resident (192 KB)", fdtd_nodes, iterations, [&]() {
        System2CoprocessorSuite::AcousticSubdomainStep_AVX512(
            p_next.data(), p_curr.data(), p_prev.data(), abs_mask.data(), 343.0f, 0.0001f, 0.05f
        );
    }));

    // K4: SO(3) Lie Camera Injector
    alignas(64) float base_vp[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    alignas(64) CameraMatrixPayload target_slot{};

    metrics.push_back(profile_kernel("K4: SO(3) Lie Camera Injector", "Taylor Series 4th-Order", 1.0, iterations, [&]() {
        System2CoprocessorSuite::InjectCameraMatrix_AVX512(
            base_vp, 12.5f, -8.3f, 0.00833f, 0.0022f, &target_slot
        );
    }));

    // K5: Chromatic PBD Cloth Solver (24,000 Edges)
    const size_t v_count = 10000;
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
        p1_groups[g].resize(6000, 0);
        p2_groups[g].resize(6000, 1);
        rest_groups[g].resize(6000, 1.0f);
        chromatic_sets[g] = {p1_groups[g].data(), p2_groups[g].data(), rest_groups[g].data(), 6000};
    }

    metrics.push_back(profile_kernel("K5: Chromatic PBD Cloth [6-CORE MT]", "6 Physical Cores (24k Edges)", e_count, iterations, [&]() {
        System2CoprocessorSuite::RelaxChromaticMesh_AVX512_MT(mesh, chromatic_sets, pool);
    }));

    // K6: Signed VNNI UI (1080p Stencil)
    const int ui_w = 1920, ui_h = 1080;
    const size_t ui_pixels = ui_w * ui_h;
    AlignedBuffer<uint8_t, 64> st_full(ui_pixels, 0);
    AlignedBuffer<uint8_t, 64> sub_coords(ui_pixels * 4, 32);
    AlignedBuffer<uint8_t, 64> dst_ui(ui_pixels * 4, 0);

    metrics.push_back(profile_kernel("K6: Signed VNNI UI [6-CORE MT]", "6 Cores + Sparse Skip (1080p)", ui_pixels, iterations, [&]() {
        System2CoprocessorSuite::MorphologicalSDFUIComposition_AVX512_MT(
            st_full.data(), sub_coords.data(), dst_ui.data(), ui_w, ui_h, pool
        );
    }));

    // Print Summary Table
    std::cout << "\n=============================================== [6-CORE MULTI-THREADED PROFILE MATRIX] ===========================================\n"
              << std::left << std::setw(38) << "Kernel Component"
              << std::setw(35) << "Hardware Mapping / Sizing"
              << std::right << std::setw(15) << "Throughput"
              << std::setw(15) << "Latency (6C)"
              << std::setw(15) << "Rate (FPS)\n"
              << "----------------------------------------------------------------------------------------------------------------------------------\n";

    double total_mt_suite_us = 0.0;
    for (const auto& m : metrics) {
        std::cout << std::left << std::setw(38) << m.name
                  << std::setw(35) << m.port_info
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(13) << m.throughput_m << " M/s"
                  << std::setw(12) << m.latency_us << " us"
                  << std::setw(15) << m.fps << "\n";

        total_mt_suite_us += m.latency_us;
    }

    std::cout << "==================================================================================================================================\n";
    std::cout << "\033[1;33m[HOLISTIC SYSTEM 2 PIPELINE SUMMARY]\033[0m\n"
              << "  * Total Combined 6-Core Coprocessor Pass : " << std::fixed << std::setprecision(2) << total_mt_suite_us << " us (" << (total_mt_suite_us / 1000.0) << " ms)\n"
              << "  * 60Hz  Frame Budget Duty Cycle          : " << std::setprecision(1) << (total_mt_suite_us / 16666.6) * 100.0 << "%\n"
              << "  * 144Hz Frame Budget Duty Cycle          : " << std::setprecision(1) << (total_mt_suite_us / 6944.4) * 100.0 << "%\n"
              << "==================================================================================================================================\n";

    return 0;
}
