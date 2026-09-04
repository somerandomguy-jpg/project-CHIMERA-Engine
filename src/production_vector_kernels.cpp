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

#include "production_vector_kernels.hpp"
#include <algorithm>
#include <cstring>

namespace avx512 {

void execute_hiz_moc_culling_and_dispatch(
    const BoundingBoxPacket16* __restrict packets,
    size_t num_packets,
    const HiZTileHierarchy& __restrict hiz,
    float scale_x, float scale_y,
    VkDrawIndexedIndirectCommand* __restrict out_indirect_stream,
    uint32_t* __restrict out_visible_count)
{
    const __m512 v_sx = _mm512_set1_ps(scale_x);
    const __m512 v_sy = _mm512_set1_ps(scale_y);
    const __m512 v_z_scale = _mm512_set1_ps(65535.0f);
    const __m512 v_zero = _mm512_setzero_ps();

    uint32_t total_visible = 0;

    for (size_t p = 0; p < num_packets; ++p) {
        const auto& packet = packets[p];

        const __m512 min_x = _mm512_load_ps(packet.min_x);
        const __m512 min_y = _mm512_load_ps(packet.min_y);
        const __m512 min_z = _mm512_load_ps(packet.min_z);
        const __m512 max_z = _mm512_load_ps(packet.max_z);

        const __m512i tile_min_x = _mm512_cvt_roundps_epi32(_mm512_mul_ps(min_x, v_sx), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        const __m512i tile_min_y = _mm512_cvt_roundps_epi32(_mm512_mul_ps(min_y, v_sy), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        const __m512i znear_u16  = _mm512_cvt_roundps_epi32(_mm512_mul_ps(min_z, v_z_scale), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

        const __mmask16 m_frustum = _mm512_cmp_ps_mask(max_z, v_zero, _CMP_GT_OQ) &
                                    _mm512_cmp_ps_mask(min_z, _mm512_set1_ps(1000.0f), _CMP_LT_OQ);

        alignas(64) int32_t arr_znear[16];
        alignas(64) int32_t arr_tx_min[16], arr_ty_min[16];
        _mm512_store_si512(arr_znear, znear_u16);
        _mm512_store_si512(arr_tx_min, tile_min_x);
        _mm512_store_si512(arr_ty_min, tile_min_y);

        uint16_t hiz_vis_bits = 0;

        #pragma GCC unroll 16
        for (int i = 0; i < 16; ++i) {
            int tx = std::clamp(arr_tx_min[i], 0, static_cast<int>(HiZTileHierarchy::MIP0_W - 1));
            int ty = std::clamp(arr_ty_min[i], 0, static_cast<int>(HiZTileHierarchy::MIP0_H - 1));
            uint16_t node_max_z = hiz.mip0[ty * HiZTileHierarchy::MIP0_W + tx];

            if (static_cast<uint16_t>(arr_znear[i]) <= node_max_z + 16) {
                hiz_vis_bits |= (1 << i);
            }
        }

        const __mmask16 final_visibility = static_cast<__mmask16>(hiz_vis_bits) & m_frustum;

        if (final_visibility != 0) {
            const __m512i v_draw_ids = _mm512_load_si512(packet.draw_ids);
            alignas(64) uint32_t visible_ids[16];
            _mm512_mask_compressstoreu_epi32(visible_ids, final_visibility, v_draw_ids);
            const uint32_t count = _mm_popcnt_u32(static_cast<uint32_t>(final_visibility));

            for (uint32_t v = 0; v < count; ++v) {
                out_indirect_stream[total_visible + v] = {
                    .indexCount = 384,
                    .instanceCount = 1,
                    .firstIndex = 0,
                    .vertexOffset = static_cast<int32_t>(visible_ids[v] * 256),
                    .firstInstance = visible_ids[v]
                };
            }
            total_visible += count;
        }
    }
    *out_visible_count = total_visible;
}

void execute_separable_fwht16_denoise_540p(
    const int32_t* __restrict noisy_radiance_in,
    const int32_t* __restrict normal_grad_mod_in,
    int32_t* __restrict clean_radiance_out,
    int32_t base_lambda,
    int32_t alpha_gradient_scale,
    size_t width, size_t height)
{
    const __m512i v_lambda_base = _mm512_set1_epi32(base_lambda);
    const __m512i v_alpha_scale = _mm512_set1_epi32(alpha_gradient_scale);
    const __m512i v_zero        = _mm512_setzero_si512();

    const __m512i v_perm_transpose = _mm512_setr_epi32(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
    );

    for (size_t y = 0; y < height; y += 4) {
        for (size_t x = 0; x < width; x += 4) {
            __m512i r0 = _mm512_loadu_si512(noisy_radiance_in + (y + 0) * width + x);
            __m512i r1 = _mm512_loadu_si512(noisy_radiance_in + (y + 1) * width + x);
            __m512i r2 = _mm512_loadu_si512(noisy_radiance_in + (y + 2) * width + x);
            __m512i r3 = _mm512_loadu_si512(noisy_radiance_in + (y + 3) * width + x);

            __m512i v_block = _mm512_mask_blend_epi32(0x00F0, r0,
                              _mm512_mask_blend_epi32(0x0F00, r1,
                              _mm512_mask_blend_epi32(0xF000, r2, r3)));

            __m512i s0 = _mm512_add_epi32(v_block, _mm512_shuffle_epi32(v_block, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2, 3, 0, 1))));
            __m512i d0 = _mm512_sub_epi32(v_block, _mm512_shuffle_epi32(v_block, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2, 3, 0, 1))));
            __m512i stg1 = _mm512_mask_blend_epi32(0xAAAA, s0, d0);

            __m512i s1 = _mm512_add_epi32(stg1, _mm512_shuffle_epi32(stg1, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1, 0, 3, 2))));
            __m512i d1 = _mm512_sub_epi32(stg1, _mm512_shuffle_epi32(stg1, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1, 0, 3, 2))));
            __m512i h_transformed = _mm512_mask_blend_epi32(0xCCCC, s1, d1);

            __m512i v_transposed = _mm512_permutexvar_epi32(v_perm_transpose, h_transformed);

            __m512i vs0 = _mm512_add_epi32(v_transposed, _mm512_shuffle_epi32(v_transposed, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2, 3, 0, 1))));
            __m512i vd0 = _mm512_sub_epi32(v_transposed, _mm512_shuffle_epi32(v_transposed, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2, 3, 0, 1))));
            __m512i vstg1 = _mm512_mask_blend_epi32(0xAAAA, vs0, vd0);

            __m512i vs1 = _mm512_add_epi32(vstg1, _mm512_shuffle_epi32(vstg1, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1, 0, 3, 2))));
            __m512i vd1 = _mm512_sub_epi32(vstg1, _mm512_shuffle_epi32(vstg1, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1, 0, 3, 2))));
            __m512i fwd_2d_fwht = _mm512_mask_blend_epi32(0xCCCC, vs1, vd1);

            __m512i v_norm_grad = _mm512_loadu_si512(normal_grad_mod_in + y * width + x);
            __m512i v_lambda = _mm512_max_epi32(v_zero, _mm512_sub_epi32(v_lambda_base, _mm512_mullo_epi32(v_norm_grad, v_alpha_scale)));
            v_lambda = _mm512_mask_blend_epi32(0x0001, v_lambda, v_zero);

            __m512i v_sign = _mm512_srai_epi32(fwd_2d_fwht, 31);
            __m512i v_abs  = _mm512_abs_epi32(fwd_2d_fwht);
            __m512i v_shrunk_abs = _mm512_max_epi32(v_zero, _mm512_sub_epi32(v_abs, v_lambda));
            __m512i v_denoised_coeff = _mm512_sub_epi32(_mm512_xor_si512(v_shrunk_abs, v_sign), v_sign);

            __m512i inv_s0 = _mm512_add_epi32(v_denoised_coeff, _mm512_shuffle_epi32(v_denoised_coeff, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2, 3, 0, 1))));
            __m512i inv_d0 = _mm512_sub_epi32(v_denoised_coeff, _mm512_shuffle_epi32(v_denoised_coeff, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2, 3, 0, 1))));
            __m512i inv_stg1 = _mm512_mask_blend_epi32(0xAAAA, inv_s0, inv_d0);

            __m512i inv_s1 = _mm512_add_epi32(inv_stg1, _mm512_shuffle_epi32(inv_stg1, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1, 0, 3, 2))));
            __m512i inv_d1 = _mm512_sub_epi32(inv_stg1, _mm512_shuffle_epi32(inv_stg1, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1, 0, 3, 2))));
            __m512i inv_h = _mm512_mask_blend_epi32(0xCCCC, inv_s1, inv_d1);

            __m512i inv_t = _mm512_permutexvar_epi32(v_perm_transpose, inv_h);

            __m512i fin_s0 = _mm512_add_epi32(inv_t, _mm512_shuffle_epi32(inv_t, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2, 3, 0, 1))));
            __m512i fin_d0 = _mm512_sub_epi32(inv_t, _mm512_shuffle_epi32(inv_t, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2, 3, 0, 1))));
            __m512i fin_stg1 = _mm512_mask_blend_epi32(0xAAAA, fin_s0, fin_d0);

            __m512i fin_s1 = _mm512_add_epi32(fin_stg1, _mm512_shuffle_epi32(fin_stg1, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1, 0, 3, 2))));
            __m512i fin_d1 = _mm512_sub_epi32(fin_stg1, _mm512_shuffle_epi32(fin_stg1, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1, 0, 3, 2))));
            __m512i recovered_block = _mm512_mask_blend_epi32(0xCCCC, fin_s1, fin_d1);

            __m512i out_pixels = _mm512_srai_epi32(recovered_block, 4);

            alignas(64) int32_t raw_out[16];
            _mm512_store_si512(raw_out, out_pixels);
            for (size_t row = 0; row < 4; ++row) {
                std::memcpy(clean_radiance_out + (y + row) * width + x, raw_out + row * 4, 4 * sizeof(int32_t));
            }
        }
    }
}

// ================================================================================================
// MODULE 03: In-Place Multiplierless CSD Gamut Remapper & PCHIP Tone Engine
// ================================================================================================
void execute_csd_gamut_and_pchip_tonemap_in_place_540p(
    uint32_t* __restrict inout_rgba_540p,
    size_t width, size_t height)
{
    const size_t total_pixels = width * height;
    const __m512i v_zero = _mm512_setzero_si512();

    for (size_t i = 0; i < total_pixels; i += 16) {
        const __m512i rgba_in = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(inout_rgba_540p + i));

        // Unpack R, G, B channels from rendered 3D scene
        const __m512i r_in = _mm512_and_si512(rgba_in, _mm512_set1_epi32(0xFF));
        const __m512i g_in = _mm512_and_si512(_mm512_srli_epi32(rgba_in, 8), _mm512_set1_epi32(0xFF));
        const __m512i b_in = _mm512_and_si512(_mm512_srli_epi32(rgba_in, 16), _mm512_set1_epi32(0xFF));

        // 1. RGB -> YCoCg in registers (Port 0/1)
        const __m512i y  = _mm512_srai_epi32(_mm512_add_epi32(r_in, _mm512_add_epi32(_mm512_slli_epi32(g_in, 1), b_in)), 2);
        const __m512i co = _mm512_srai_epi32(_mm512_sub_epi32(_mm512_slli_epi32(r_in, 1), _mm512_slli_epi32(b_in, 1)), 2);
        const __m512i cg = _mm512_srai_epi32(_mm512_sub_epi32(_mm512_slli_epi32(g_in, 1), _mm512_add_epi32(r_in, b_in)), 2);

        // 2. CSD Matrix: YCoCg -> Rec.2020 Display Gamut
        const __m512i r_2020 = _mm512_add_epi32(y,
                               _mm512_add_epi32(_mm512_sub_epi32(_mm512_slli_epi32(co, 1), _mm512_srai_epi32(co, 2)),
                                                _mm512_srai_epi32(cg, 3)));
        const __m512i g_2020 = _mm512_sub_epi32(_mm512_add_epi32(y, _mm512_srai_epi32(cg, 2)),
                               _mm512_add_epi32(_mm512_slli_epi32(cg, 1), _mm512_srai_epi32(co, 4)));
        const __m512i b_2020 = _mm512_add_epi32(_mm512_sub_epi32(y, _mm512_slli_epi32(co, 1)),
                               _mm512_add_epi32(_mm512_srai_epi32(co, 2),
                               _mm512_sub_epi32(_mm512_srai_epi32(cg, 2), _mm512_slli_epi32(cg, 1))));

        // 3. Monotonic PCHIP Curve Tone Mapping
        auto eval_pchip = [&](__m512i val) -> __m512i {
            val = _mm512_max_epi32(val, v_zero);
            const __m512 v_f = _mm512_cvtepi32_ps(val);
            const __m512 t = _mm512_min_ps(_mm512_mul_ps(v_f, _mm512_set1_ps(1.0f / 255.0f)), _mm512_set1_ps(1.0f));
            const __m512 oetf = _mm512_mul_ps(t, _mm512_fnmadd_ps(t, _mm512_set1_ps(0.75f), _mm512_set1_ps(1.75f)));
            return _mm512_cvtps_epi32(_mm512_mul_ps(oetf, _mm512_set1_ps(255.0f)));
        };

        const __m512i r_out = eval_pchip(r_2020);
        const __m512i g_out = eval_pchip(g_2020);
        const __m512i b_out = eval_pchip(b_2020);
        const __m512i a_out = _mm512_set1_epi32(255);

        const __m512i rgba_final = _mm512_or_epi32(
            _mm512_or_epi32(r_out, _mm512_slli_epi32(g_out, 8)),
            _mm512_or_epi32(_mm512_slli_epi32(b_out, 16), _mm512_slli_epi32(a_out, 24))
        );

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(inout_rgba_540p + i), rgba_final);
    }
}

void execute_wavelet_dual_lobe_gtao_540p(
    const uint16_t* __restrict depth_in,
    uint8_t* __restrict ao_vis_out,
    uint16_t flat_threshold,
    size_t width, size_t height)
{
    const size_t total_pixels = width * height;
    const __m512i v_flat_tau = _mm512_set1_epi16(flat_threshold);
    const __m512i v_one_q15  = _mm512_set1_epi16(32767);

    for (size_t i = 0; i < total_pixels; i += 32) {
        const __m512i z_prev = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(depth_in + i - 1));
        const __m512i z_next = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(depth_in + i + 1));

        const __m512i grad_h = _mm512_abs_epi16(_mm512_sub_epi16(z_next, z_prev));
        const __mmask32 m_skip = _mm512_cmple_epu16_mask(grad_h, v_flat_tau);

        __m512i cos_theta = _mm512_subs_epu16(v_one_q15, _mm512_slli_epi16(grad_h, 3));

        __m512i u1 = _mm512_mulhrs_epi16(cos_theta, cos_theta);
        __m512i u2 = _mm512_mulhrs_epi16(u1, u1);
        __m512i u3 = _mm512_mulhrs_epi16(u2, u2);
        __m512i spec_lobe = _mm512_mulhrs_epi16(u3, u3);

        __m512i diff_term = _mm512_mulhrs_epi16(cos_theta, _mm512_set1_epi16(19660));
        __m512i spec_term = _mm512_mulhrs_epi16(spec_lobe, _mm512_set1_epi16(13107));
        __m512i ao_val_16 = _mm512_adds_epu16(diff_term, spec_term);

        __m512i ao_u8 = _mm512_srli_epi16(ao_val_16, 7);
        ao_u8 = _mm512_mask_blend_epi16(m_skip, ao_u8, _mm512_set1_epi16(255));

        __m256i packed_32b = _mm512_cvtepi16_epi8(ao_u8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ao_vis_out + i), packed_32b);
    }
}

void execute_clamped_variance_taa_rectifier(
    const uint32_t* __restrict curr_color_in,
    uint32_t* __restrict inout_hist_color,
    uint32_t* __restrict rectified_color_out,
    uint8_t alpha_blend_weight,
    size_t width, size_t height)
{
    const size_t total_pixels = width * height;
    const __m512i v_zero = _mm512_setzero_si512();

    const uint32_t blend_pair = (static_cast<uint32_t>(128 - alpha_blend_weight) << 8) | alpha_blend_weight;
    const __m512i v_vnni_weights = _mm512_set1_epi32(blend_pair);

    for (size_t i = 0; i < total_pixels; i += 16) {
        const __m512i curr = _mm512_loadu_si512(curr_color_in + i);
        const __m512i hist = _mm512_loadu_si512(inout_hist_color + i);
        const __m512i c_left  = _mm512_loadu_si512(curr_color_in + i - 1);
        const __m512i c_right = _mm512_loadu_si512(curr_color_in + i + 1);

        __m512i curr_r = _mm512_and_si512(curr, _mm512_set1_epi32(0xFF));
        __m512i hist_r = _mm512_and_si512(hist, _mm512_set1_epi32(0xFF));

        __m512i sum_r = _mm512_add_epi32(curr_r, _mm512_add_epi32(
                        _mm512_and_si512(c_left, _mm512_set1_epi32(0xFF)),
                        _mm512_and_si512(c_right, _mm512_set1_epi32(0xFF))));
        __m512i mu_r = _mm512_srai_epi32(_mm512_mullo_epi32(sum_r, _mm512_set1_epi32(10923)), 15);

        __m512i exp_c2 = _mm512_madd_epi16(curr_r, curr_r);
        __m512i mu_sq = _mm512_mullo_epi32(mu_r, mu_r);
        __m512i sigma_sq = _mm512_max_epi32(_mm512_sub_epi32(exp_c2, mu_sq), v_zero);

        __m512 sigma_f = _mm512_mul_ps(_mm512_cvtepi32_ps(sigma_sq), _mm512_rsqrt14_ps(_mm512_cvtepi32_ps(sigma_sq)));
        __m512i sigma_r = _mm512_cvtps_epi32(sigma_f);

        __m512i sigma_1_5 = _mm512_add_epi32(sigma_r, _mm512_srai_epi32(sigma_r, 1));
        __m512i min_bound = _mm512_max_epi32(v_zero, _mm512_sub_epi32(mu_r, sigma_1_5));
        __m512i max_bound = _mm512_min_epi32(_mm512_set1_epi32(255), _mm512_add_epi32(mu_r, sigma_1_5));

        __m512i clamped_hist_r = _mm512_min_epi32(max_bound, _mm512_max_epi32(min_bound, hist_r));

        __m512i pair_r = _mm512_or_si512(curr_r, _mm512_slli_epi32(clamped_hist_r, 8));
        __m512i blended_r = _mm512_srai_epi32(_mm512_dpbusd_epi32(v_zero, pair_r, v_vnni_weights), 7);

        __m512i final_rgba = _mm512_or_si512(blended_r, _mm512_set1_epi32(0xFF000000));
        _mm512_storeu_si512(rectified_color_out + i, final_rgba);
        _mm512_storeu_si512(inout_hist_color + i, final_rgba);
    }
}

void execute_oct16_sh_irradiance_solver_540p(
    const uint16_t* __restrict oct16_normals_in,
    const SH9ColorCoefficients& __restrict sh_ambient_probe,
    uint32_t* __restrict irradiance_rgb_out,
    size_t width, size_t height)
{
    const size_t total_pixels = width * height;
    const __m512 v_one  = _mm512_set1_ps(1.0f);
    const __m512 v_zero = _mm512_setzero_ps();
    const __m512 v_inv127 = _mm512_set1_ps(1.0f / 127.0f);

    const __m512 l00_r = _mm512_set1_ps(sh_ambient_probe.r[0]);
    const __m512 l10_r = _mm512_set1_ps(sh_ambient_probe.r[2]);
    const __m512 l11_r = _mm512_set1_ps(sh_ambient_probe.r[3]);
    const __m512 l1m1_r= _mm512_set1_ps(sh_ambient_probe.r[1]);
    const __m512 l20_r = _mm512_set1_ps(sh_ambient_probe.r[6]);
    const __m512 l21_r = _mm512_set1_ps(sh_ambient_probe.r[7]);
    const __m512 l22_r = _mm512_set1_ps(sh_ambient_probe.r[8]);
    const __m512 l2m1_r= _mm512_set1_ps(sh_ambient_probe.r[5]);
    const __m512 l2m2_r= _mm512_set1_ps(sh_ambient_probe.r[4]);

    for (size_t i = 0; i < total_pixels; i += 16) {
        const __m256i raw_oct = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(oct16_normals_in + i));
        const __m512i oct_32  = _mm512_cvtepu16_epi32(raw_oct);

        const __m512i u_s8 = _mm512_srai_epi32(_mm512_slli_epi32(oct_32, 24), 24);
        const __m512i v_s8 = _mm512_srai_epi32(_mm512_slli_epi32(oct_32, 16), 24);

        const __m512 u = _mm512_mul_ps(_mm512_cvtepi32_ps(u_s8), v_inv127);
        const __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(v_s8), v_inv127);

        const __m512 abs_u = _mm512_abs_ps(u);
        const __m512 abs_v = _mm512_abs_ps(v);
        const __m512 z_orig = _mm512_sub_ps(v_one, _mm512_add_ps(abs_u, abs_v));
        const __mmask16 m_wrap = _mm512_cmp_ps_mask(z_orig, v_zero, _CMP_LT_OQ);

        const __m512 sign_u = _mm512_sub_ps(v_zero, _mm512_and_ps(u, _mm512_set1_ps(-0.0f)));
        const __m512 sign_v = _mm512_sub_ps(v_zero, _mm512_and_ps(v, _mm512_set1_ps(-0.0f)));

        const __m512 x = _mm512_mask_blend_ps(m_wrap, u, _mm512_mul_ps(_mm512_sub_ps(v_one, abs_v), sign_u));
        const __m512 y = _mm512_mask_blend_ps(m_wrap, v, _mm512_mul_ps(_mm512_sub_ps(v_one, abs_u), sign_v));
        const __m512 z = z_orig;

        const __m512 len_sq = _mm512_fmadd_ps(z, z, _mm512_fmadd_ps(y, y, _mm512_mul_ps(x, x)));
        const __m512 inv_len = _mm512_rsqrt14_ps(len_sq);

        const __m512 nx = _mm512_mul_ps(x, inv_len);
        const __m512 ny = _mm512_mul_ps(y, inv_len);
        const __m512 nz = _mm512_mul_ps(z, inv_len);

        const __m512 x2 = _mm512_mul_ps(nx, nx);
        const __m512 y2 = _mm512_mul_ps(ny, ny);
        const __m512 z2 = _mm512_mul_ps(nz, nz);
        const __m512 xy = _mm512_mul_ps(nx, ny);
        const __m512 yz = _mm512_mul_ps(ny, nz);
        const __m512 xz = _mm512_mul_ps(nx, nz);

        __m512 irr_r = _mm512_mul_ps(l00_r, _mm512_set1_ps(0.886227f));
        irr_r = _mm512_fmadd_ps(l20_r, _mm512_fmsub_ps(z2, _mm512_set1_ps(0.743125f), _mm512_set1_ps(0.247708f)), irr_r);
        irr_r = _mm512_fmadd_ps(l22_r, _mm512_mul_ps(_mm512_sub_ps(x2, y2), _mm512_set1_ps(0.429043f)), irr_r);
        irr_r = _mm512_fmadd_ps(l2m2_r, _mm512_mul_ps(xy, _mm512_set1_ps(0.858086f)), irr_r);
        irr_r = _mm512_fmadd_ps(l21_r,  _mm512_mul_ps(xz, _mm512_set1_ps(0.858086f)), irr_r);
        irr_r = _mm512_fmadd_ps(l2m1_r, _mm512_mul_ps(yz, _mm512_set1_ps(0.858086f)), irr_r);
        irr_r = _mm512_fmadd_ps(l11_r,  _mm512_mul_ps(nx, _mm512_set1_ps(1.023327f)), irr_r);
        irr_r = _mm512_fmadd_ps(l1m1_r, _mm512_mul_ps(ny, _mm512_set1_ps(1.023327f)), irr_r);
        irr_r = _mm512_fmadd_ps(l10_r,  _mm512_mul_ps(nz, _mm512_set1_ps(1.023327f)), irr_r);

        irr_r = _mm512_max_ps(irr_r, v_zero);
        const __m512i r_u32 = _mm512_cvtps_epi32(_mm512_mul_ps(irr_r, _mm512_set1_ps(255.0f)));
        const __m512i rgba = _mm512_or_si512(r_u32, _mm512_set1_epi32(0xFF000000));
        _mm512_storeu_si512(irradiance_rgb_out + i, rgba);
    }
}

} // namespace avx512
