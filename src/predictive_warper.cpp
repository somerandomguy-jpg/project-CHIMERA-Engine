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

#include "predictive_warper.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

void PredictiveFrameWarper::warp_predictive_affine(
    const uint32_t* __restrict src, uint32_t* __restrict dst,
    int width, int height,
    float delta_mouse_x, float delta_mouse_y,
    CorePinnedThreadPool& pool) 
{
    // First-order camera rotational projection (Negative input lag scaling)
    const int int_dx = std::clamp(static_cast<int>(std::round(-delta_mouse_x * 0.75f)), -48, 48);
    const int int_dy = std::clamp(static_cast<int>(std::round(-delta_mouse_y * 0.75f)), -48, 48);

    const __m512i offset_x = _mm512_set1_epi32(int_dx);
    const __m512i lane_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i min_x = _mm512_setzero_si512();
    const __m512i max_x = _mm512_set1_epi32(width - 1);

    pool.parallel_for(height, [=](size_t y_idx, size_t) {
        int y = static_cast<int>(y_idx);
        int src_y = std::clamp(y + int_dy, 0, height - 1);
        const uint32_t* src_row = src + src_y * width;
        uint32_t* dst_row = dst + y * width;

        for (int x = 0; x + 16 <= width; x += 16) {
            __m512i cur_x = _mm512_add_epi32(_mm512_set1_epi32(x), lane_idx);
            __m512i target_x = _mm512_min_epi32(_mm512_max_epi32(_mm512_add_epi32(cur_x, offset_x), min_x), max_x);

            __m512i warped = _mm512_i32gather_epi32(target_x, src_row, 4);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row + x), warped);
        }
    });
    _mm_sfence();
}
