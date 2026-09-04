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
#ifndef IN_ENGINE_COPROCESSOR_HPP
#define IN_ENGINE_COPROCESSOR_HPP

#include <cstdint>
#include "thread_pool.hpp"

class InEngineAvx512Coprocessor {
public:
    // In-Engine G-Buffer Super-Resolution & Motion Reprojection Pass (Replaces GPU Compute Shaders)
    static void dispatch_in_engine_fsr2_override(
        const uint32_t* __restrict color_720p,
        const float* __restrict depth_720p,
        const float* __restrict motion_vectors_720p, // Normalized Screen-Space Velocity (Vx, Vy)
        const uint32_t* __restrict prev_1080p_history,
        uint32_t* __restrict out_1080p_target,
        float jitter_x, float jitter_y,
        int src_w, int src_h, int dst_w, int dst_h,
        CorePinnedThreadPool& pool);
};

#endif
