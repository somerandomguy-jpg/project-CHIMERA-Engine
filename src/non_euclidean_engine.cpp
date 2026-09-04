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

#include "non_euclidean_engine.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace avx512 {

NonEuclideanVectorAccelerator::NonEuclideanVectorAccelerator() {
    m_volumetric_grid.resize(1);
}

void NonEuclideanVectorAccelerator::execute_pga_shadow_sieve(
    const PGALinePacket16* __restrict shadow_rays, size_t num_packets,
    const float* __restrict occluder_planes, size_t num_planes,
    uint16_t* __restrict out_occlusion_masks)
{
    const __m512 v_zero = _mm512_setzero_ps();

    for (size_t p = 0; p < num_packets; ++p) {
        const __m512 dx = _mm512_load_ps(shadow_rays[p].dx);
        const __m512 dy = _mm512_load_ps(shadow_rays[p].dy);
        const __m512 dz = _mm512_load_ps(shadow_rays[p].dz);

        const __m512 mx = _mm512_load_ps(shadow_rays[p].mx);
        const __m512 my = _mm512_load_ps(shadow_rays[p].my);
        const __m512 mz = _mm512_load_ps(shadow_rays[p].mz);

        __mmask16 occluded_mask = 0x0000;

        for (size_t pl = 0; pl < num_planes; ++pl) {
            const float* plane = occluder_planes + pl * 4;
            const __m512 px = _mm512_set1_ps(plane[0]);
            const __m512 py = _mm512_set1_ps(plane[1]);
            const __m512 pz = _mm512_set1_ps(plane[2]);
            const __m512 pd = _mm512_set1_ps(plane[3]);

            const __m512 d_dot_n = _mm512_fmadd_ps(dz, pz, _mm512_fmadd_ps(dy, py, _mm512_mul_ps(dx, px)));
            const __m512 mxn_x = _mm512_fmsub_ps(my, pz, _mm512_mul_ps(mz, py));
            const __m512 mxn_y = _mm512_fmsub_ps(mz, px, _mm512_mul_ps(mx, pz));
            const __m512 mxn_z = _mm512_fmsub_ps(mx, py, _mm512_mul_ps(my, px));

            const __m512 pt_x = _mm512_fmadd_ps(pd, dx, mxn_x);
            const __m512 pt_y = _mm512_fmadd_ps(pd, dy, mxn_y);
            const __m512 pt_z = _mm512_fmadd_ps(pd, dz, mxn_z);

            __m512 pt_len_sq = _mm512_mul_ps(pt_x, pt_x);
            pt_len_sq = _mm512_fmadd_ps(pt_y, pt_y, pt_len_sq);
            pt_len_sq = _mm512_fmadd_ps(pt_z, pt_z, pt_len_sq);

            const __mmask16 hit = _mm512_cmp_ps_mask(d_dot_n, v_zero, _CMP_NEQ_OQ) &
                                  _mm512_cmp_ps_mask(pt_len_sq, v_zero, _CMP_GE_OQ);
            occluded_mask |= hit;
        }

        out_occlusion_masks[p] = static_cast<uint16_t>(occluded_mask);
    }
}

inline void NonEuclideanVectorAccelerator::evaluate_minkowski_point_lights_16x(
    const __m512 px, const __m512 py, const __m512 pz,
    const MinkowskiLightSource& light,
    __m512& out_attenuation, __mmask16& out_active_mask)
{
    const __m512 lx = _mm512_set1_ps(light.x);
    const __m512 ly = _mm512_set1_ps(light.y);
    const __m512 lz = _mm512_set1_ps(light.z);
    const __m512 lr = _mm512_set1_ps(light.radius);
    const __m512 lr_sq = _mm512_mul_ps(lr, lr);

    const __m512 dx = _mm512_sub_ps(px, lx);
    const __m512 dy = _mm512_sub_ps(py, ly);
    const __m512 dz = _mm512_sub_ps(pz, lz);

    __m512 dist_sq = _mm512_mul_ps(dx, dx);
    dist_sq = _mm512_fmadd_ps(dy, dy, dist_sq);
    dist_sq = _mm512_fmadd_ps(dz, dz, dist_sq);

    const __m512 delta_s_sq = _mm512_sub_ps(lr_sq, dist_sq);
    out_active_mask = _mm512_cmp_ps_mask(delta_s_sq, _mm512_setzero_ps(), _CMP_GT_OQ);

    const __m512 norm_interval = _mm512_max_ps(_mm512_mul_ps(delta_s_sq, _mm512_rcp14_ps(lr_sq)), _mm512_setzero_ps());
    out_attenuation = _mm512_mul_ps(norm_interval, norm_interval);
}

void NonEuclideanVectorAccelerator::execute_chandrasekhar_kalman_volumetrics(
    const MinkowskiLightSource* __restrict lights, size_t num_lights,
    float scattering_coeff, float extinction_coeff, float,
    uint32_t* __restrict,
    CorePinnedThreadPool& pool)
{
    auto& grid = m_volumetric_grid[0];

    struct KalmanVolumetricPayload {
        NonEuclideanVectorAccelerator* accel;
        VolumetricFrustumGrid* grid;
        const MinkowskiLightSource* lights;
        size_t num_lights;
        float sigma_s;
        float sigma_t;
    } payload{this, &grid, lights, num_lights, scattering_coeff, extinction_coeff};

    pool.parallel_for([](void* user_data, uint32_t thread_id, uint32_t num_threads) {
        auto* p = static_cast<KalmanVolumetricPayload*>(user_data);
        auto* grid = p->grid;

        const size_t rows_per_core = (VolumetricFrustumGrid::GRID_Y + num_threads - 1) / num_threads;
        const size_t y_start = thread_id * rows_per_core;
        const size_t y_end   = std::min(y_start + rows_per_core, VolumetricFrustumGrid::GRID_Y);

        const __m512 v_dx_16   = _mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        const __m512 v_sigma_s = _mm512_set1_ps(p->sigma_s);
        const __m512 v_sigma_t = _mm512_set1_ps(p->sigma_t);
        const __m512 v_one     = _mm512_set1_ps(1.0f);
        const __m512 v_half    = _mm512_set1_ps(0.5f);
        const __m512 v_eight   = _mm512_set1_ps(8.0f);
        const __m512 v_point2  = _mm512_set1_ps(0.2f);

        for (size_t ty = y_start; ty < y_end; ++ty) {
            __m512 state_accum_r = _mm512_setzero_ps();
            __m512 state_accum_g = _mm512_setzero_ps();
            __m512 state_accum_b = _mm512_setzero_ps();
            __m512 state_transmittance = _mm512_set1_ps(1.0f);

            for (size_t kz = 0; kz < VolumetricFrustumGrid::GRID_Z; ++kz) {
                const size_t voxel_row_idx = kz * 256 + ty * 16;

                const float delta_z = 0.4f * std::exp(0.06f * static_cast<float>(kz));
                const __m512 v_dz = _mm512_set1_ps(delta_z);

                const __m512 opt_depth = _mm512_mul_ps(v_sigma_t, v_dz);
                const __m512 transition_f = _mm512_sub_ps(v_one, _mm512_mul_ps(opt_depth, _mm512_fnmadd_ps(opt_depth, v_half, v_one)));

                const __m512 vx_pos = _mm512_mul_ps(_mm512_sub_ps(v_dx_16, v_eight), _mm512_mul_ps(v_point2, v_dz));
                const __m512 vy_pos = _mm512_mul_ps(_mm512_sub_ps(_mm512_set1_ps(static_cast<float>(ty)), v_eight), _mm512_mul_ps(v_point2, v_dz));
                const __m512 vz_pos = _mm512_set1_ps(static_cast<float>(kz) * delta_z + 1.0f);

                __m512 in_scatter_r = _mm512_setzero_ps();
                __m512 in_scatter_g = _mm512_setzero_ps();
                __m512 in_scatter_b = _mm512_setzero_ps();

                for (size_t l = 0; l < p->num_lights; ++l) {
                    const auto& light = p->lights[l];
                    __m512 atten; __mmask16 mask;
                    p->accel->evaluate_minkowski_point_lights_16x(vx_pos, vy_pos, vz_pos, light, atten, mask);

                    if (mask != 0) {
                        const __m512 lr = _mm512_mul_ps(_mm512_set1_ps(light.color_r * light.intensity), atten);
                        const __m512 lg = _mm512_mul_ps(_mm512_set1_ps(light.color_g * light.intensity), atten);
                        const __m512 lb = _mm512_mul_ps(_mm512_set1_ps(light.color_b * light.intensity), atten);

                        in_scatter_r = _mm512_add_ps(in_scatter_r, lr);
                        in_scatter_g = _mm512_add_ps(in_scatter_g, lg);
                        in_scatter_b = _mm512_add_ps(in_scatter_b, lb);
                    }
                }

                const __m512 scatter_gain = _mm512_mul_ps(_mm512_mul_ps(v_sigma_s, state_transmittance), v_dz);
                state_accum_r = _mm512_fmadd_ps(in_scatter_r, scatter_gain, _mm512_mul_ps(transition_f, state_accum_r));
                state_accum_g = _mm512_fmadd_ps(in_scatter_g, scatter_gain, _mm512_mul_ps(transition_f, state_accum_g));
                state_accum_b = _mm512_fmadd_ps(in_scatter_b, scatter_gain, _mm512_mul_ps(transition_f, state_accum_b));
                state_transmittance = _mm512_mul_ps(state_transmittance, transition_f);

                _mm512_store_ps(grid->integrated_light_r + voxel_row_idx, state_accum_r);
                _mm512_store_ps(grid->integrated_light_g + voxel_row_idx, state_accum_g);
                _mm512_store_ps(grid->integrated_light_b + voxel_row_idx, state_accum_b);
                _mm512_store_ps(grid->integrated_transmittance + voxel_row_idx, state_transmittance);
            }
        }
    }, &payload);
}

} // namespace avx512
