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

struct alignas(64) CPUMeshTriangle {
    float x0, y0, z0; float nx0, ny0, nz0;
    float x1, y1, z1; float nx1, ny1, nz1;
    float x2, y2, z2; float nx2, ny2, nz2;
    float r, g, b;     // PBR Material Base Albedo [0.0, 1.0]
    float roughness;   // Roughness [0.0 = Wet Mirror, 1.0 = Rough Thatch]
    uint32_t is_hair{0};
};

struct alignas(64) TransformedTriangle {
    float x0, y0, z0, inv_w0;
    float x1, y1, z1, inv_w1;
    float x2, y2, z2, inv_w2;
    float nx0, ny0, nz0;
    float nx1, ny1, nz1;
    float nx2, ny2, nz2;
    float a01, b01, c01;
    float a12, b12, c12;
    float a20, b20, c20;
    float inv_area;
    float r, g, b;
    float roughness;
    uint32_t is_hair;
};

struct alignas(64) RasterTile16x16 {
    static constexpr size_t TILE_SIZE = 16;
    static constexpr size_t PIXEL_COUNT = 256;

    alignas(64) float depth[PIXEL_COUNT];
    alignas(64) uint32_t rgba[PIXEL_COUNT];
    uint32_t tri_count{0};
    uint32_t tri_indices[2048];
};

class SoftwareRendererAVX512 {
public:
    SoftwareRendererAVX512(size_t render_w = 960, size_t render_h = 540);
    ~SoftwareRendererAVX512() = default;

    void render_scene_to_framebuffer(
        const CPUMeshTriangle* triangles, size_t num_triangles,
        const float* view_proj_matrix,
        uint32_t* dst_color_540p,
        CorePinnedThreadPool& pool);

    size_t get_render_w() const noexcept { return m_render_w; }
    size_t get_render_h() const noexcept { return m_render_h; }

private:
    size_t m_render_w;
    size_t m_render_h;
    size_t m_tiles_x;
    size_t m_tiles_y;
    size_t m_total_tiles;

    AlignedBuffer<TransformedTriangle, 64> m_transformed_triangles;
    AlignedBuffer<RasterTile16x16, 64>     m_tiles;
};

} // namespace avx512
