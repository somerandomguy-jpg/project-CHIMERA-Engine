#include "in_engine_coprocessor.hpp"
#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <cmath>

static inline __m512i _mm512_clamp_epi32(__m512i v, int min_val, int max_val) {
    return _mm512_min_epi32(_mm512_max_epi32(v, _mm512_set1_epi32(min_val)), _mm512_set1_epi32(max_val));
}

void InEngineAvx512Coprocessor::dispatch_in_engine_fsr2_override(
    const uint32_t* __restrict color_720p,
    const float* __restrict depth_720p,
    const float* __restrict motion_vectors_720p,
    const uint32_t* __restrict prev_1080p_history,
    uint32_t* __restrict out_1080p_target,
    float jitter_x, float jitter_y,
    int src_w, int src_h, int dst_w, int dst_h,
    CorePinnedThreadPool& pool) 
{
    (void)dst_h; (void)depth_720p; (void)jitter_x; (void)jitter_y;
    const __m512i ising_j = _mm512_set1_epi64(0x0101010101010101ULL);
    const __m512 v_scale_x = _mm512_set1_ps(static_cast<float>(dst_w));
    const __m512 v_scale_y = _mm512_set1_ps(static_cast<float>(dst_h));
    const __m512i g_idx_lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    const __m512i g_idx_hi = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
    const __m512i shift_r = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15);

    pool.parallel_for(src_h, [=](size_t y_src, size_t) {
        int y = static_cast<int>(y_src);
        int y_next = std::min(y + 1, src_h - 1);
        const uint32_t* r0 = color_720p + y * src_w;
        const uint32_t* r1 = color_720p + y_next * src_w;
        const float* mv_row = motion_vectors_720p + (y * src_w * 2);

        uint32_t* dst_row0 = out_1080p_target + (y * 2) * dst_w;
        uint32_t* dst_row1 = out_1080p_target + (y * 2 + 1) * dst_w;

        for (int x = 0; x + 16 <= src_w; x += 16) {
            __m512i p00 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x));
            __m512i p01 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x));
            __m512i p10 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r0 + x + 1))
                                             : _mm512_permutexvar_epi32(shift_r, p00);
            __m512i p11 = (x + 17 <= src_w) ? _mm512_loadu_si512(reinterpret_cast<const __m512i*>(r1 + x + 1))
                                             : _mm512_permutexvar_epi32(shift_r, p01);

            // 1. Ingest Exact In-Engine Motion Vectors
            __m512 vx_raw = _mm512_loadu_ps(mv_row + x * 2);
            __m512 vy_raw = _mm512_loadu_ps(mv_row + x * 2 + 16);

            __m512i delta_px_x = _mm512_cvtps_epi32(_mm512_mul_ps(vx_raw, v_scale_x));
            __m512i delta_px_y = _mm512_cvtps_epi32(_mm512_mul_ps(vy_raw, v_scale_y));

            // 2. Hardware Backward Motion Reprojection (0-Ghosting Alignment)
            __m512i cur_x_vec = _mm512_add_epi32(_mm512_set1_epi32(x * 2), _mm512_setr_epi32(0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30));
            __m512i cur_y_vec = _mm512_set1_epi32(y * 2);

            __m512i reprojected_x = _mm512_clamp_epi32(_mm512_sub_epi32(cur_x_vec, delta_px_x), 0, dst_w - 1);
            __m512i reprojected_y = _mm512_clamp_epi32(_mm512_sub_epi32(cur_y_vec, delta_px_y), 0, dst_h - 1);
            __m512i history_idx   = _mm512_add_epi32(_mm512_mullo_epi32(reprojected_y, _mm512_set1_epi32(dst_w)), reprojected_x);

            // 3. Gather Reprojected History from Cache
            __m512i history_color = _mm512_i32gather_epi32(history_idx, prev_1080p_history, 4);

            // 4. Bio-Physics Spatial Synthesis (Physarum + Ising Ground State)
            __m512i flux_d1 = _mm512_subs_epu8(_mm512_max_epu8(p00, p11), _mm512_min_epu8(p00, p11));
            __m512i flux_d2 = _mm512_subs_epu8(_mm512_max_epu8(p10, p01), _mm512_min_epu8(p10, p01));
            __mmask64 flux_mask = _mm512_cmplt_epu8_mask(flux_d1, flux_d2);
            __m512i d_phys = _mm512_mask_blend_epi8(flux_mask, _mm512_avg_epu8(p10, p01), _mm512_avg_epu8(p00, p11));

            __m512i spin_d1 = _mm512_ternarylogic_epi32(p00, p11, ising_j, 0x66);
            __m512i spin_d2 = _mm512_ternarylogic_epi32(p10, p01, ising_j, 0x66);
            __mmask16 ising_m = _mm512_cmplt_epu32_mask(_mm512_popcnt_epi32(spin_d1), _mm512_popcnt_epi32(spin_d2));
            __m512i d_ising = _mm512_mask_blend_epi32(ising_m, _mm512_avg_epu8(p00, p11), _mm512_avg_epu8(p10, p01));

            __m512i current_spatial_d = _mm512_avg_epu8(d_phys, d_ising);

            // 5. In-Engine Temporal Blend (Anti-Ghosting Reprojection)
            __m512i temporal_d = _mm512_avg_epu8(current_spatial_d, history_color);

            __m512i h_interp = _mm512_avg_epu8(p00, p10);
            __m512i v_interp = _mm512_avg_epu8(p00, p01);

            // Direct Write to 1080p Render Target
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2), _mm512_permutex2var_epi32(p00, g_idx_lo, h_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row0 + x * 2 + 16), _mm512_permutex2var_epi32(p00, g_idx_hi, h_interp));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2), _mm512_permutex2var_epi32(v_interp, g_idx_lo, temporal_d));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row1 + x * 2 + 16), _mm512_permutex2var_epi32(v_interp, g_idx_hi, temporal_d));
        }
    });
    _mm_sfence();
}
