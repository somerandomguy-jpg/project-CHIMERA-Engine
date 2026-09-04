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

#include <immintrin.h>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include "thread_pool.hpp"

// ================================================================================================
// SYSTEM 2 VERIFIED PORTFOLIO DATA STRUCTURES (K1 - K6)
// ================================================================================================

struct BoundingSpheresSoA {
    const float* cx;
    const float* cy;
    const float* cz;
    const float* cr;
};

struct DrawCallDescriptor {
    uint32_t pso_id{0};
    uint32_t pipeline_id{0};
    uint32_t mesh_id{0};
    uint32_t material_id{0};
    uint32_t index_count{0};
    uint32_t first_index{0};
    int32_t  vertex_offset{0};
    uint32_t push_constant_size{0};
    const uint8_t* push_constant_payload{nullptr};
};

struct alignas(64) CameraMatrixPayload {
    float view_proj[16];
};

struct MeshParticlesSoA {
    float* px;
    float* py;
    float* pz;
    float* inv_m;
    size_t count;
};

struct ChromaticEdgeSet {
    const uint32_t* p1;
    const uint32_t* p2;
    const float* rest_length;
    size_t count;
};

class System2CoprocessorSuite {
public:
    static void FrustumCullAndSerializeIndirect_AVX512(
        const BoundingSpheresSoA& spheres, size_t count, const float planes[6][4],
        const uint32_t* idx_counts, const uint32_t* first_indices, const int32_t* vertex_offsets,
        uint8_t* dst_indirect_bar, uint32_t* total_draws_out);

    static void FrustumCullAndSerializeIndirect_AVX512(
        const BoundingSpheresSoA& spheres, size_t count, const float* planes,
        const uint32_t* idx_counts, const uint32_t* first_indices, const int32_t* vertex_offsets,
        uint8_t* dst_indirect_bar, uint32_t* total_draws_out);

    static void CoalesceDrawCommands_AVX512(
        const DrawCallDescriptor* draws, size_t count,
        uint32_t* batch_offsets, uint32_t* batch_counts, uint32_t* total_batches_out);

    static void AcousticSubdomainStep_AVX512(
        float* p_next, const float* p_curr, const float* p_prev, const uint8_t* abs_mask,
        float c, float dt, float dx);

    static void InjectCameraMatrix_AVX512(
        const float* base_vp, float pitch_rate, float yaw_rate, float dt_pred, float fov_scale,
        CameraMatrixPayload* target_slot);

    static void RelaxChromaticMesh_AVX512(MeshParticlesSoA& mesh, const ChromaticEdgeSet& edges);
    static void RelaxChromaticMesh_AVX512_MT(
        MeshParticlesSoA& mesh, const std::vector<ChromaticEdgeSet>& chromatic_sets, CorePinnedThreadPool& pool);

    static void MorphologicalSDFUIComposition_AVX512(
        const uint8_t* stencil, const uint8_t* r, const uint8_t* g, const uint8_t* sub_coords,
        uint8_t* dst_rgba, size_t num_pixels);
    static void MorphologicalSDFUIComposition_AVX512_MT(
        const uint8_t* stencil, const uint8_t* sub_coords, uint8_t* dst_ui,
        int width, int height, CorePinnedThreadPool& pool);
};

// ================================================================================================
// FRONTIER COPROCESSOR DATA STRUCTURES (K8 - K11)
// ================================================================================================

namespace avx512 {

struct alignas(64) ClusterAABBArray {
    static constexpr size_t NUM_CLUSTERS = 8192;
    static constexpr size_t NUM_PACKETS  = NUM_CLUSTERS / 16;

    float min_x[NUM_PACKETS][16];
    float min_y[NUM_PACKETS][16];
    float min_z[NUM_PACKETS][16];
    float max_x[NUM_PACKETS][16];
    float max_y[NUM_PACKETS][16];
    float max_z[NUM_PACKETS][16];
};

struct alignas(16) PointLightViewSpace {
    float x, y, z, radius;
    float color_r, color_g, color_b, intensity;
};

struct alignas(64) ClusterLightBitmaskTable {
    static constexpr size_t MAX_LIGHTS = 256;
    static constexpr size_t MASKS_PER_CLUSTER = MAX_LIGHTS / 64;
    uint64_t light_masks[8192][MASKS_PER_CLUSTER];
};

struct alignas(64) SH9ColorPacket {
    float c[9][3][16];
};

struct alignas(64) HDRPixelBuffer {
    std::vector<float> r;
    std::vector<float> g;
    std::vector<float> b;
    size_t width{1920};
    size_t height{1080};
};

void avx512_sieve_lights_for_cluster_range(
    const ClusterAABBArray& __restrict clusters,
    const PointLightViewSpace* __restrict lights,
    size_t num_lights,
    ClusterLightBitmaskTable& __restrict output_table,
    size_t packet_start,
    size_t packet_end);

void avx512_wigner_ggx_compensate_row(
    const float* __restrict n_prev_x, const float* __restrict n_prev_y, const float* __restrict n_prev_z,
    const float* __restrict n_curr_x, const float* __restrict n_curr_y, const float* __restrict n_curr_z,
    const float* __restrict n_next_x, const float* __restrict n_next_y, const float* __restrict n_next_z,
    const float* __restrict roughness_in,
    float* __restrict roughness_out,
    size_t width);

__m512 avx512_aces_fitted_ps(__m512 x);

void avx512_tonemap_row(
    const float* __restrict r_in, const float* __restrict g_in, const float* __restrict b_in,
    uint32_t* __restrict ldr_out, size_t width);

class CoprocessorEngine {
public:
    CoprocessorEngine();
    ~CoprocessorEngine() = default;

    void initialize_clusters(float fov_y, float aspect, float z_near, float z_far);

    void execute_frame(
        const PointLightViewSpace* lights, size_t num_lights,
        const float* normal_gbuffer, float* roughness_gbuffer,
        const HDRPixelBuffer& hdr_in, uint32_t* ldr_out,
        size_t width, size_t height);

    const ClusterLightBitmaskTable& get_light_table() const noexcept { return m_light_table; }

private:
    CorePinnedThreadPool m_pool{6};

    alignas(64) ClusterAABBArray         m_clusters;
    alignas(64) ClusterLightBitmaskTable m_light_table;
    alignas(64) std::vector<SH9ColorPacket> m_probe_grid;
};

} // namespace avx512
