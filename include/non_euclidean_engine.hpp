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
#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include "thread_pool.hpp"
#include "aligned_buffer.hpp"

namespace avx512 {

struct alignas(64) PGALinePacket16 {
    float dx[16], dy[16], dz[16];
    float mx[16], my[16], mz[16];
};

struct alignas(16) MinkowskiLightSource {
    float x, y, z;
    float radius;
    float color_r, color_g, color_b;
    float intensity;
};

struct alignas(64) VolumetricFrustumGrid {
    static constexpr size_t GRID_X = 16;
    static constexpr size_t GRID_Y = 16;
    static constexpr size_t GRID_Z = 32;
    static constexpr size_t TOTAL_VOXELS = GRID_X * GRID_Y * GRID_Z;

    float in_scattering_r[TOTAL_VOXELS];
    float in_scattering_g[TOTAL_VOXELS];
    float in_scattering_b[TOTAL_VOXELS];
    float transmittance[TOTAL_VOXELS];
    float integrated_light_r[TOTAL_VOXELS];
    float integrated_light_g[TOTAL_VOXELS];
    float integrated_light_b[TOTAL_VOXELS];
    float integrated_transmittance[TOTAL_VOXELS];
};

class NonEuclideanVectorAccelerator {
public:
    NonEuclideanVectorAccelerator();
    ~NonEuclideanVectorAccelerator() = default;

    void execute_pga_shadow_sieve(
        const PGALinePacket16* shadow_rays, size_t num_packets,
        const float* occluder_planes, size_t num_planes,
        uint16_t* out_occlusion_masks);

    void execute_chandrasekhar_kalman_volumetrics(
        const MinkowskiLightSource* lights, size_t num_lights,
        float scattering_coeff, float extinction_coeff, float time_sec,
        uint32_t* dst_volumetric_fog_1080p,
        CorePinnedThreadPool& pool);

    void evaluate_minkowski_point_lights_16x(
        const __m512 px, const __m512 py, const __m512 pz,
        const MinkowskiLightSource& light,
        __m512& out_attenuation, __mmask16& out_active_mask);

private:
    AlignedBuffer<VolumetricFrustumGrid, 64> m_volumetric_grid;
};

} // namespace avx512
