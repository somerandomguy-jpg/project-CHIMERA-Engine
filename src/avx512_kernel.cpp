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

#include "avx512_kernel.hpp"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

const __m512i g_idx_2x_lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
const __m512i g_idx_2x_hi = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
const __m512i idx_shift_right = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15);

// ============================================================================
// [FLAGSHIP V4] QUAD-STREAM 4X UNROLLED OMNI SUPER-KERNEL (64 Pixels / Step)
// ============================================================================
void Avx512Upscaler::upscale_tiered_hyper_omni_v4(
    const uint32_t* __restrict src, uint32_t* __restrict dst,
    int src_w, int src_h, int dst_w, int dst_h,
    uint64_t frame_index, CorePinnedThreadPool& pool) 
{
    (void)dst_h;
    const __m512i v_decay = _mm512_set1_epi8(4);
    const __m512i v_thresh = _mm512_set1_epi8(28);
    const __m512i ising_j = _mm512_set1_epi64(0x0101010101010101ULL);
    const uint64_t entropy_seed = 0x1B8420910A4C2E01ULL ^ (frame_index * 0x9E3779B97F4A7C15ULL);
    const __m512i lfsr_matrix = _mm512_set1_epi64(entropy_seed);
    const __m512i one_mask = _mm512_set1_epi8(1);

    pool.parallel_for(src_h, [=](size_t y_src, size_t) {
        int y = static_cast<int>(y_src);
        int y_next = std::min(y + 1, src_h - 1);
        const uint32_t* r0 = src + y * src_w;
        const uint32_t* r1 = src + y_next * src_w;
        uint32_t* dst_row0 = dst + (y * 2) * dst_w;
        uint32_t* dst_row1 = dst + (y * 2 + 1) * dst_w;

        // 4x Quad-Stream Unrolled Loop (Processes 64 pixels / 256 bytes per iteration)
        int x = 0;
        for (; x + 64 <= src_w; x += 64) {
            // Load 4 vector streams in parallel to fully hide L1d load latencies
            __m512i p00_0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x));
            __m512i p01_0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x));
            __m512i p10_0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 1));
            __m512i p11_0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 1));

            __m512i p00_1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 16));
            __m512i p01_1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 16));
            __m512i p10_1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 17));
            __m512i p11_1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 17));

            __m512i p00_2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 32));
            __m512i p01_2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 32));
            __m512i p10_2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 33));
            __m512i p11_2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 33));

            __m512i p00_3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 48));
            __m512i p01_3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 48));
            __m512i p10_3 = (x + 65 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 49)) : _mm512_permutexvar_epi32(idx_shift_right, p00_3);
            __m512i p11_3 = (x + 65 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 49)) : _mm512_permutexvar_epi32(idx_shift_right, p01_3);

            // Micro-kernel lambda for quad-stream parallel execution
            auto process_block = [&](__m512i p00, __m512i p01, __m512i p10, __m512i p11, int out_offset) {
                __m512i h_int = _mm512_avg_epu8(p00, p10);
                __m512i v_int = _mm512_avg_epu8(p00, p01);
                __m512i d_base = _mm512_avg_epu8(h_int, _mm512_avg_epu8(p01, p11));

                __m512i h_diff = _mm512_subs_epu8(_mm512_max_epu8(p00, p10), _mm512_min_epu8(p00, p10));
                __m512i v_diff = _mm512_subs_epu8(_mm512_max_epu8(p00, p01), _mm512_min_epu8(p00, p01));
                __mmask64 spike = _mm512_cmpgt_epu8_mask(_mm512_subs_epu8(_mm512_adds_epu8(h_diff, v_diff), v_decay), v_thresh);

                __m512i flux1 = _mm512_subs_epu8(_mm512_max_epu8(p00, p11), _mm512_min_epu8(p00, p11));
                __m512i flux2 = _mm512_subs_epu8(_mm512_max_epu8(p10, p01), _mm512_min_epu8(p10, p01));
                __m512i d_phys = _mm512_mask_blend_epi8(_mm512_cmplt_epu8_mask(flux1, flux2), _mm512_avg_epu8(p10, p01), _mm512_avg_epu8(p00, p11));

                __m512i spin1 = _mm512_ternarylogic_epi32(p00, p11, ising_j, 0x66);
                __m512i spin2 = _mm512_ternarylogic_epi32(p10, p01, ising_j, 0x66);
                __mmask16 ising_m = _mm512_cmplt_epu32_mask(_mm512_popcnt_epi32(spin1), _mm512_popcnt_epi32(spin2));
                __m512i d_ising = _mm512_mask_blend_epi32(ising_m, _mm512_avg_epu8(p00, p11), _mm512_avg_epu8(p10, p01));
                __m512i d_edge = _mm512_avg_epu8(d_phys, d_ising);

                __m512i noise = _mm512_gf2p8affine_epi64_epi8(d_edge, lfsr_matrix, 0x00);
                __m512i d_dither = _mm512_subs_epu8(_mm512_adds_epu8(d_edge, _mm512_and_si512(noise, one_mask)), _mm512_and_si512(_mm512_srli_epi16(noise, 1), one_mask));
                __m512i d_final = _mm512_mask_blend_epi8(spike, d_base, d_dither);

                _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + out_offset), _mm512_permutex2var_epi32(p00, g_idx_2x_lo, h_int));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + out_offset + 16), _mm512_permutex2var_epi32(p00, g_idx_2x_hi, h_int));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + out_offset), _mm512_permutex2var_epi32(v_int, g_idx_2x_lo, d_final));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + out_offset + 16), _mm512_permutex2var_epi32(v_int, g_idx_2x_hi, d_final));
            };

            process_block(p00_0, p01_0, p10_0, p11_0, x * 2);
            process_block(p00_1, p01_1, p10_1, p11_1, (x + 16) * 2);
            process_block(p00_2, p01_2, p10_2, p11_2, (x + 32) * 2);
            process_block(p00_3, p01_3, p10_3, p11_3, (x + 48) * 2);
        }

        // Tail loop for boundary pixels
        for (; x + 16 <= src_w; x += 16) {
            __m512i p00 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x));
            __m512i p01 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x));
            __m512i p10 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 1)) : _mm512_permutexvar_epi32(idx_shift_right, p00);
            __m512i p11 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 1)) : _mm512_permutexvar_epi32(idx_shift_right, p01);

            __m512i h_int = _mm512_avg_epu8(p00, p10);
            __m512i v_int = _mm512_avg_epu8(p00, p01);
            __m512i d_base = _mm512_avg_epu8(h_int, _mm512_avg_epu8(p01, p11));

            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2), _mm512_permutex2var_epi32(p00, g_idx_2x_lo, h_int));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2 + 16), _mm512_permutex2var_epi32(p00, g_idx_2x_hi, h_int));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2), _mm512_permutex2var_epi32(v_int, g_idx_2x_lo, d_base));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2 + 16), _mm512_permutex2var_epi32(v_int, g_idx_2x_hi, d_base));
        }
    });
    _mm_sfence();
}

void Avx512Upscaler::upscale_physarum_slime_mold_2x(const uint32_t* src, uint32_t* dst, int src_w, int src_h, int dst_w, int dst_h, CorePinnedThreadPool& pool) {
    (void)dst_h;
    pool.parallel_for(src_h, [=](size_t y_src, size_t) {
        int y = static_cast<int>(y_src), y_next = std::min(y + 1, src_h - 1);
        const uint32_t* r0 = src + y * src_w; const uint32_t* r1 = src + y_next * src_w;
        uint32_t* dst_row0 = dst + (y * 2) * dst_w; uint32_t* dst_row1 = dst + (y * 2 + 1) * dst_w;

        for (int x = 0; x + 16 <= src_w; x += 16) {
            __m512i p00 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x));
            __m512i p01 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x));
            __m512i p10 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 1)) : _mm512_permutexvar_epi32(idx_shift_right, p00);
            __m512i p11 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 1)) : _mm512_permutexvar_epi32(idx_shift_right, p01);

            __m512i flux_d1 = _mm512_subs_epu8(_mm512_max_epu8(p00, p11), _mm512_min_epu8(p00, p11));
            __m512i flux_d2 = _mm512_subs_epu8(_mm512_max_epu8(p10, p01), _mm512_min_epu8(p10, p01));
            __mmask64 flux_mask = _mm512_cmplt_epu8_mask(flux_d1, flux_d2);

            __m512i vein_d1 = _mm512_avg_epu8(p00, p11), vein_d2 = _mm512_avg_epu8(p10, p01);
            __m512i d_physarum = _mm512_mask_blend_epi8(flux_mask, vein_d2, vein_d1);
            __m512i h_interp = _mm512_avg_epu8(p00, p10), v_interp = _mm512_avg_epu8(p00, p01);

            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2), _mm512_permutex2var_epi32(p00, g_idx_2x_lo, h_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2 + 16), _mm512_permutex2var_epi32(p00, g_idx_2x_hi, h_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2), _mm512_permutex2var_epi32(v_interp, g_idx_2x_lo, d_physarum));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2 + 16), _mm512_permutex2var_epi32(v_interp, g_idx_2x_hi, d_physarum));
        }
    });
    _mm_sfence();
}

void Avx512Upscaler::upscale_quantum_ising_glass_2x(const uint32_t* src, uint32_t* dst, int src_w, int src_h, int dst_w, int dst_h, CorePinnedThreadPool& pool) {
    (void)dst_h;
    const __m512i ising_j = _mm512_set1_epi64(0x0101010101010101ULL);
    pool.parallel_for(src_h, [=](size_t y_src, size_t) {
        int y = static_cast<int>(y_src), y_next = std::min(y + 1, src_h - 1);
        const uint32_t* r0 = src + y * src_w; const uint32_t* r1 = src + y_next * src_w;
        uint32_t* dst_row0 = dst + (y * 2) * dst_w; uint32_t* dst_row1 = dst + (y * 2 + 1) * dst_w;

        for (int x = 0; x + 16 <= src_w; x += 16) {
            __m512i p00 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x));
            __m512i p01 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x));
            __m512i p10 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 1)) : _mm512_permutexvar_epi32(idx_shift_right, p00);
            __m512i p11 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 1)) : _mm512_permutexvar_epi32(idx_shift_right, p01);

            __m512i spin_d1 = _mm512_ternarylogic_epi32(p00, p11, ising_j, 0x66);
            __m512i spin_d2 = _mm512_ternarylogic_epi32(p10, p01, ising_j, 0x66);
            __mmask16 ising_mask = _mm512_cmplt_epu32_mask(_mm512_popcnt_epi32(spin_d1), _mm512_popcnt_epi32(spin_d2));

            __m512i d1 = _mm512_avg_epu8(p00, p11), d2 = _mm512_avg_epu8(p10, p01);
            __m512i d_ising = _mm512_mask_blend_epi32(ising_mask, d1, d2);
            __m512i h_interp = _mm512_avg_epu8(p00, p10), v_interp = _mm512_avg_epu8(p00, p01);

            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2), _mm512_permutex2var_epi32(p00, g_idx_2x_lo, h_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2 + 16), _mm512_permutex2var_epi32(p00, g_idx_2x_hi, h_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2), _mm512_permutex2var_epi32(v_interp, g_idx_2x_lo, d_ising));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2 + 16), _mm512_permutex2var_epi32(v_interp, g_idx_2x_hi, d_ising));
        }
    });
    _mm_sfence();
}

void Avx512Upscaler::upscale_neuromorphic_lif_2x(const uint32_t* src, uint32_t* dst, int src_w, int src_h, int dst_w, int dst_h, CorePinnedThreadPool& pool) {
    (void)dst_h;
    const __m512i v_decay = _mm512_set1_epi8(4), v_thresh = _mm512_set1_epi8(32);
    pool.parallel_for(src_h, [=](size_t y_src, size_t) {
        int y = static_cast<int>(y_src), y_next = std::min(y + 1, src_h - 1);
        const uint32_t* r0 = src + y * src_w; const uint32_t* r1 = src + y_next * src_w;
        uint32_t* dst_row0 = dst + (y * 2) * dst_w; uint32_t* dst_row1 = dst + (y * 2 + 1) * dst_w;

        for (int x = 0; x + 16 <= src_w; x += 16) {
            __m512i p00 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x));
            __m512i p01 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x));
            __m512i p10 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 1)) : _mm512_permutexvar_epi32(idx_shift_right, p00);
            __m512i p11 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 1)) : _mm512_permutexvar_epi32(idx_shift_right, p01);

            __m512i h_diff = _mm512_subs_epu8(_mm512_max_epu8(p00, p10), _mm512_min_epu8(p00, p10));
            __m512i v_diff = _mm512_subs_epu8(_mm512_max_epu8(p00, p01), _mm512_min_epu8(p00, p01));
            __mmask64 spike_mask = _mm512_cmpgt_epu8_mask(_mm512_subs_epu8(_mm512_adds_epu8(h_diff, v_diff), v_decay), v_thresh);

            __m512i h_interp = _mm512_avg_epu8(p00, p10), v_interp = _mm512_avg_epu8(p00, p01);
            __m512i d_smooth = _mm512_avg_epu8(h_interp, _mm512_avg_epu8(p01, p11));
            __m512i d_final = _mm512_mask_blend_epi8(spike_mask, d_smooth, _mm512_subs_epu8(_mm512_adds_epu8(h_interp, v_interp), d_smooth));

            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2), _mm512_permutex2var_epi32(p00, g_idx_2x_lo, h_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2 + 16), _mm512_permutex2var_epi32(p00, g_idx_2x_hi, h_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2), _mm512_permutex2var_epi32(v_interp, g_idx_2x_lo, d_final));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2 + 16), _mm512_permutex2var_epi32(v_interp, g_idx_2x_hi, d_final));
        }
    });
    _mm_sfence();
}

void Avx512Upscaler::upscale_gfni_galois_recon_2x(const uint32_t* src, uint32_t* dst, int src_w, int src_h, int dst_w, int dst_h, CorePinnedThreadPool& pool) {
    (void)dst_h;
    const __m512i matrix_k = _mm512_set1_epi64(0x8040201008040201ULL);
    const __m512i mask_rb = _mm512_set1_epi32(0x00FF00FF), mask_ga = _mm512_set1_epi32(0xFF00FF00);

    pool.parallel_for(src_h, [=](size_t y_src, size_t) {
        int y = static_cast<int>(y_src), y_next = std::min(y + 1, src_h - 1);
        const uint32_t* r0 = src + y * src_w; const uint32_t* r1 = src + y_next * src_w;
        uint32_t* dst_row0 = dst + (y * 2) * dst_w; uint32_t* dst_row1 = dst + (y * 2 + 1) * dst_w;

        for (int x = 0; x + 16 <= src_w; x += 16) {
            __m512i p00 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x));
            __m512i p01 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x));
            __m512i p10 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 1)) : _mm512_permutexvar_epi32(idx_shift_right, p00);
            __m512i p11 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 1)) : _mm512_permutexvar_epi32(idx_shift_right, p01);

            __m512i p00_rb = _mm512_and_si512(p00, mask_rb), p00_ga = _mm512_and_si512(p00, mask_ga);
            __m512i p11_rb = _mm512_and_si512(p11, mask_rb), p11_ga = _mm512_and_si512(p11, mask_ga);
            __m512i gf_rb = _mm512_gf2p8affine_epi64_epi8(_mm512_avg_epu8(p00_rb, p11_rb), matrix_k, 0x00);
            __m512i gf_ga = _mm512_gf2p8affine_epi64_epi8(_mm512_avg_epu8(p00_ga, p11_ga), matrix_k, 0x00);

            __m512i d_interp = _mm512_or_si512(_mm512_and_si512(gf_rb, mask_rb), _mm512_and_si512(gf_ga, mask_ga));
            __m512i h_interp = _mm512_avg_epu8(p00, p10), v_interp = _mm512_avg_epu8(p00, p01);

            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2), _mm512_permutex2var_epi32(p00, g_idx_2x_lo, h_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2 + 16), _mm512_permutex2var_epi32(p00, g_idx_2x_hi, h_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2), _mm512_permutex2var_epi32(v_interp, g_idx_2x_lo, d_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2 + 16), _mm512_permutex2var_epi32(v_interp, g_idx_2x_hi, d_interp));
        }
    });
    _mm_sfence();
}

void Avx512Upscaler::upscale_tiered_hyper_omni_v3(
    const uint32_t* src,
    [[maybe_unused]] const uint32_t* prev_frame,
    uint32_t* dst,
    int src_w, int src_h, int dst_w, int dst_h,
    uint64_t frame_index,
    CorePinnedThreadPool& pool) {
    upscale_tiered_hyper_omni_v4(src, dst, src_w, src_h, dst_w, dst_h, frame_index, pool);
}
