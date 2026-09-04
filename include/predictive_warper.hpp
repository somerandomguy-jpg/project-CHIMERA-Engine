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
#ifndef PREDICTIVE_WARPER_HPP
#define PREDICTIVE_WARPER_HPP

#include <cstdint>
#include "thread_pool.hpp"

class PredictiveFrameWarper {
public:
    // Generates an intermediate extrapolated sub-frame in < 35 microseconds
    static void warp_predictive_affine(
        const uint32_t* __restrict src, uint32_t* __restrict dst,
        int width, int height,
        float delta_mouse_x, float delta_mouse_y,
        CorePinnedThreadPool& pool);
};
#endif
