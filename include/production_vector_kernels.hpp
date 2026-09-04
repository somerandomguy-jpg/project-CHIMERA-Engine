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

#include "thread_pool.hpp"
#include "aligned_buffer.hpp"

namespace avx512 {

// 1. Hi-Z MOC Occlusion Culler
struct alignas(64) BoundingBoxPacket16 {
    float min_x[16], min_y[16], min_z[16];
    float max_x[16], max_y[16], max_z[16];
    uint32_t draw_ids[16];
};

struct alignas(64) HiZTileHierarchy {
    static constexpr size_t MIP0_W = 160, MIP0_H = 90;
    uint16_t mip0[MIP0_W * MIP0_H];
};

void execute_hiz_moc_culling_and_dispatch(
    const BoundingBoxPacket16* __restrict packets,
    size_t num_packets,
    const HiZTileHierarchy& __restrict hiz,
    float scale_x, float scale_y,
    VkDrawIndexedIndirectCommand* __restrict out_indirect_stream,
    uint32_t* __restrict out_visible_count);

// 2. FWHT-16 Specular Denoiser
void execute_separable_fwht16_denoise_540p(
    const int32_t* __restrict noisy_radiance_in,
    const int32_t* __restrict normal_grad_mod_in,
    int32_t* __restrict clean_radiance_out,
    int32_t base_lambda,
    int32_t alpha_gradient_scale,
    size_t width, size_t height);

// 3. In-Place Multiplierless CSD Gamut Remapper & PCHIP Tone Engine
void execute_csd_gamut_and_pchip_tonemap_in_place_540p(
    uint32_t* __restrict inout_rgba_540p,
    size_t width, size_t height);

// 4. Wavelet Dual-Lobe GTAO
void execute_wavelet_dual_lobe_gtao_540p(
    const uint16_t* __restrict depth_in,
    uint8_t* __restrict ao_vis_out,
    uint16_t flat_threshold,
    size_t width, size_t height);

// 5. Clamped Variance TAA Rectifier
void execute_clamped_variance_taa_rectifier(
    const uint32_t* __restrict curr_color_in,
    uint32_t* __restrict inout_hist_color,
    uint32_t* __restrict rectified_color_out,
    uint8_t alpha_blend_weight,
    size_t width, size_t height);

// 6. Oct16 Spherical Harmonics Solver
struct alignas(64) SH9ColorCoefficients {
    float r[9], g[9], b[9];
};

void execute_oct16_sh_irradiance_solver_540p(
    const uint16_t* __restrict oct16_normals_in,
    const SH9ColorCoefficients& __restrict sh_ambient_probe,
    uint32_t* __restrict irradiance_rgb_out,
    size_t width, size_t height);

} // namespace avx512
