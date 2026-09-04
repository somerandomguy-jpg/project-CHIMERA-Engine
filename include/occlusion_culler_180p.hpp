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
#include "thread_pool.hpp"
#include "aligned_buffer.hpp"

namespace avx512 {

struct alignas(64) BoundingBoxSoA16 {
    float min_x[16], min_y[16], min_z[16];
    float max_x[16], max_y[16], max_z[16];
};

struct alignas(64) DepthBuffer180p {
    static constexpr size_t WIDTH  = 160;
    static constexpr size_t HEIGHT = 90;
    static constexpr size_t PIXEL_COUNT = WIDTH * HEIGHT; // 14,400 pixels = 57.6 KB (L1d Cache)

    alignas(64) float depth[PIXEL_COUNT];
};

class OcclusionCuller180p {
public:
    OcclusionCuller180p();
    ~OcclusionCuller180p() = default;

    void clear_depth(float clear_z = 1.0f);

    void test_occlusion_16x(
        const BoundingBoxSoA16& boxes,
        float fov_x, float fov_y,
        __mmask16& out_visible_mask);

    void cull_scene_objects(
        const BoundingBoxSoA16* object_boxes, size_t num_packets,
        float fov_x, float fov_y,
        uint16_t* out_visibility_masks);

    DepthBuffer180p& get_buffer() noexcept { return m_depth_180p[0]; }

private:
    CorePinnedThreadPool m_pool{6};
    AlignedBuffer<DepthBuffer180p, 64> m_depth_180p;
};

} // namespace avx512
