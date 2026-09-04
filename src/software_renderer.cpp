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

#include "software_renderer.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>

namespace avx512 {

SoftwareRendererAVX512::SoftwareRendererAVX512(size_t render_w, size_t render_h)
    : m_render_w(render_w), m_render_h(render_h)
{
    m_tiles_x = (m_render_w + RasterTile16x16::TILE_SIZE - 1) / RasterTile16x16::TILE_SIZE;
    m_tiles_y = (m_render_h + RasterTile16x16::TILE_SIZE - 1) / RasterTile16x16::TILE_SIZE;
    m_total_tiles = m_tiles_x * m_tiles_y;

    m_transformed_triangles.resize(131072);
    m_tiles.resize(m_total_tiles);
}

void avx512_rasterize_tile_scanlines(
    const TransformedTriangle* __restrict triangles,
    RasterTile16x16& __restrict tile,
    size_t tile_x_base, size_t tile_y_base)
{
    const __m512 v_dx_16 = _mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    // Dynamic Morning Sunrise Sky Gradient (Fading from Golden Horizon to Morning Blue Zenith)
    const float horizon_y = 270.0f; // Screen horizon line
    for (size_t row = 0; row < 16; ++row) {
        float py = static_cast<float>(tile_y_base + row);
        float t = std::clamp((horizon_y - py) / 270.0f, 0.0f, 1.0f); // 0 = horizon, 1 = zenith

        // Horizon: Warm Golden Orange (#D4843A) -> Zenith: Morning Blue (#4A688C)
        float sky_r = (1.0f - t) * 0.83f + t * 0.29f;
        float sky_g = (1.0f - t) * 0.52f + t * 0.41f;
        float sky_b = (1.0f - t) * 0.23f + t * 0.55f;

        uint32_t r_u8 = static_cast<uint32_t>(sky_r * 255.0f);
        uint32_t g_u8 = static_cast<uint32_t>(sky_g * 255.0f);
        uint32_t b_u8 = static_cast<uint32_t>(sky_b * 255.0f);
        uint32_t sky_color = (r_u8) | (g_u8 << 8) | (b_u8 << 16) | (0xFF000000);

        _mm512_store_ps(tile.depth + row * 16, _mm512_set1_ps(1.0f));
        _mm512_store_si512(reinterpret_cast<__m512i*>(tile.rgba + row * 16), _mm512_set1_epi32(sky_color));
    }

    // Low Morning Sun Light L1 = normalize(0.12, 0.38, 0.91)
    const __m512 l1x = _mm512_set1_ps(0.121f), l1y = _mm512_set1_ps(0.384f), l1z = _mm512_set1_ps(0.915f);
    // Cool Sky Ambient L2 = normalize(-0.2, 0.9, -0.2)
    const __m512 l2x = _mm512_set1_ps(-0.213f), l2y = _mm512_set1_ps(0.954f), l2z = _mm512_set1_ps(-0.213f);

    const __m512 h1x = _mm512_set1_ps(0.063f), h1y = _mm512_set1_ps(0.201f), h1z = _mm512_set1_ps(0.978f);

    const __m512 v_one  = _mm512_set1_ps(1.0f);
    const __m512 v_zero = _mm512_setzero_ps();
    const __m512 v_255  = _mm512_set1_ps(255.0f);
    const __m512i a_u32 = _mm512_set1_epi32(255);

    for (uint32_t t = 0; t < tile.tri_count; ++t) {
        const auto& tri = triangles[tile.tri_indices[t]];

        const __m512 mat_r = _mm512_set1_ps(tri.r);
        const __m512 mat_g = _mm512_set1_ps(tri.g);
        const __m512 mat_b = _mm512_set1_ps(tri.b);
        const __m512 mat_rough = _mm512_set1_ps(tri.roughness);

        for (size_t y = 0; y < 16; ++y) {
            const float py_val = static_cast<float>(tile_y_base + y) + 0.5f;
            const float px_base = static_cast<float>(tile_x_base) + 0.5f;

            const __m512 py = _mm512_set1_ps(py_val);
            const __m512 px = _mm512_add_ps(_mm512_set1_ps(px_base), v_dx_16);

            const __m512 e01 = _mm512_fmadd_ps(_mm512_set1_ps(tri.a01), px, _mm512_fmadd_ps(_mm512_set1_ps(tri.b01), py, _mm512_set1_ps(tri.c01)));
            const __m512 e12 = _mm512_fmadd_ps(_mm512_set1_ps(tri.a12), px, _mm512_fmadd_ps(_mm512_set1_ps(tri.b12), py, _mm512_set1_ps(tri.c12)));
            const __m512 e20 = _mm512_fmadd_ps(_mm512_set1_ps(tri.a20), px, _mm512_fmadd_ps(_mm512_set1_ps(tri.b20), py, _mm512_set1_ps(tri.c20)));

            const __mmask16 inside = _mm512_cmp_ps_mask(e01, v_zero, _CMP_GE_OQ) &
                                     _mm512_cmp_ps_mask(e12, v_zero, _CMP_GE_OQ) &
                                     _mm512_cmp_ps_mask(e20, v_zero, _CMP_GE_OQ);

            if (inside == 0) continue;

            const __m512 inv_a = _mm512_set1_ps(tri.inv_area);
            const __m512 w0 = _mm512_mul_ps(e12, inv_a);
            const __m512 w1 = _mm512_mul_ps(e20, inv_a);
            const __m512 w2 = _mm512_mul_ps(e01, inv_a);

            const __m512 interp_z = _mm512_fmadd_ps(w0, _mm512_set1_ps(tri.z0),
                                    _mm512_fmadd_ps(w1, _mm512_set1_ps(tri.z1),
                                    _mm512_mul_ps(w2, _mm512_set1_ps(tri.z2))));

            const size_t row_offset = y * 16;
            const __m512 current_depth = _mm512_load_ps(tile.depth + row_offset);
            const __mmask16 depth_pass = _mm512_mask_cmp_ps_mask(inside, interp_z, current_depth, _CMP_LT_OQ);

            if (depth_pass != 0) {
                _mm512_mask_store_ps(tile.depth + row_offset, depth_pass, interp_z);

                const __m512 nx = _mm512_fmadd_ps(w0, _mm512_set1_ps(tri.nx0), _mm512_fmadd_ps(w1, _mm512_set1_ps(tri.nx1), _mm512_mul_ps(w2, _mm512_set1_ps(tri.nx2))));
                const __m512 ny = _mm512_fmadd_ps(w0, _mm512_set1_ps(tri.ny0), _mm512_fmadd_ps(w1, _mm512_set1_ps(tri.ny1), _mm512_mul_ps(w2, _mm512_set1_ps(tri.ny2))));
                const __m512 nz = _mm512_fmadd_ps(w0, _mm512_set1_ps(tri.nz0), _mm512_fmadd_ps(w1, _mm512_set1_ps(tri.nz1), _mm512_mul_ps(w2, _mm512_set1_ps(tri.nz2))));

                const __m512 len_sq = _mm512_fmadd_ps(nz, nz, _mm512_fmadd_ps(ny, ny, _mm512_mul_ps(nx, nx)));
                const __m512 inv_len = _mm512_rsqrt14_ps(_mm512_max_ps(len_sq, _mm512_set1_ps(1e-6f)));

                const __m512 norm_x = _mm512_mul_ps(nx, inv_len);
                const __m512 norm_y = _mm512_mul_ps(ny, inv_len);
                const __m512 norm_z = _mm512_mul_ps(nz, inv_len);

                // Sun Diffuse + Sky Ambient
                const __m512 n_dot_l1 = _mm512_max_ps(_mm512_fmadd_ps(norm_z, l1z, _mm512_fmadd_ps(norm_y, l1y, _mm512_mul_ps(norm_x, l1x))), v_zero);
                const __m512 n_dot_l2 = _mm512_max_ps(_mm512_fmadd_ps(norm_z, l2z, _mm512_fmadd_ps(norm_y, l2y, _mm512_mul_ps(norm_x, l2x))), v_zero);

                // Specular GGX / Blinn-Phong (Wet Mud & Roof Sheen)
                const __m512 n_dot_h = _mm512_max_ps(_mm512_fmadd_ps(norm_z, h1z, _mm512_fmadd_ps(norm_y, h1y, _mm512_mul_ps(norm_x, h1x))), v_zero);
                __m512 spec = _mm512_mul_ps(n_dot_h, n_dot_h);
                spec = _mm512_mul_ps(spec, spec);
                spec = _mm512_mul_ps(spec, spec);
                spec = _mm512_mul_ps(spec, spec);

                const __m512 spec_strength = _mm512_mul_ps(_mm512_sub_ps(v_one, mat_rough), _mm512_set1_ps(0.85f));
                const __m512 ambient = _mm512_set1_ps(0.18f);

                __m512 lit_r = _mm512_fmadd_ps(mat_r, _mm512_add_ps(n_dot_l1, ambient), _mm512_mul_ps(spec, spec_strength));
                __m512 lit_g = _mm512_fmadd_ps(mat_g, _mm512_add_ps(n_dot_l1, ambient), _mm512_mul_ps(spec, spec_strength));
                __m512 lit_b = _mm512_fmadd_ps(mat_b, _mm512_add_ps(_mm512_mul_ps(n_dot_l2, _mm512_set1_ps(0.35f)), ambient), _mm512_mul_ps(spec, spec_strength));

                lit_r = _mm512_min_ps(lit_r, v_one);
                lit_g = _mm512_min_ps(lit_g, v_one);
                lit_b = _mm512_min_ps(lit_b, v_one);

                const __m512i r_u32 = _mm512_cvttps_epi32(_mm512_mul_ps(lit_r, v_255));
                const __m512i g_u32 = _mm512_cvttps_epi32(_mm512_mul_ps(lit_g, v_255));
                const __m512i b_u32 = _mm512_cvttps_epi32(_mm512_mul_ps(lit_b, v_255));

                const __m512i final_rgba = _mm512_or_epi32(
                    _mm512_or_epi32(r_u32, _mm512_slli_epi32(g_u32, 8)),
                    _mm512_or_epi32(_mm512_slli_epi32(b_u32, 16), _mm512_slli_epi32(a_u32, 24))
                );

                _mm512_mask_store_epi32(reinterpret_cast<__m512i*>(tile.rgba + row_offset), depth_pass, final_rgba);
            }
        }
    }
}

void SoftwareRendererAVX512::render_scene_to_framebuffer(
    const CPUMeshTriangle* triangles, size_t num_triangles,
    const float* vp,
    uint32_t* dst_color_540p,
    CorePinnedThreadPool& pool)
{
    const float half_w = m_render_w * 0.5f;
    const float half_h = m_render_h * 0.5f;

    for (size_t t = 0; t < m_total_tiles; ++t) {
        m_tiles[t].tri_count = 0;
    }

    size_t visible_triangles = 0;

    for (size_t i = 0; i < num_triangles; ++i) {
        const auto& in = triangles[i];

        auto project_v = [&](float x, float y, float z, float& sx, float& sy, float& sz, float& iw) {
            float cx = x * vp[0] + y * vp[4] + z * vp[8]  + vp[12];
            float cy = x * vp[1] + y * vp[5] + z * vp[9]  + vp[13];
            float cz = x * vp[2] + y * vp[6] + z * vp[10] + vp[14];
            float cw = x * vp[3] + y * vp[7] + z * vp[11] + vp[15];

            if (cw <= 0.05f) { iw = -1.0f; return; }
            iw = 1.0f / cw;
            sx = (cx * iw) * half_w + half_w;
            sy = (-cy * iw) * half_h + half_h;
            sz = cz * iw;
        };

        TransformedTriangle tri;
        project_v(in.x0, in.y0, in.z0, tri.x0, tri.y0, tri.z0, tri.inv_w0);
        project_v(in.x1, in.y1, in.z1, tri.x1, tri.y1, tri.z1, tri.inv_w1);
        project_v(in.x2, in.y2, in.z2, tri.x2, tri.y2, tri.z2, tri.inv_w2);

        if (tri.inv_w0 < 0.0f || tri.inv_w1 < 0.0f || tri.inv_w2 < 0.0f) continue;

        tri.a01 = tri.y0 - tri.y1; tri.b01 = tri.x1 - tri.x0; tri.c01 = tri.x0 * tri.y1 - tri.x1 * tri.y0;
        tri.a12 = tri.y1 - tri.y2; tri.b12 = tri.x2 - tri.x1; tri.c12 = tri.x1 * tri.y2 - tri.x2 * tri.y1;
        tri.a20 = tri.y2 - tri.y0; tri.b20 = tri.x0 - tri.x2; tri.c20 = tri.x2 * tri.y0 - tri.x0 * tri.y2;

        float area = tri.c01 + tri.c12 + tri.c20;
        if (area == 0.0f) continue;

        if (area < 0.0f && in.is_hair) {
            std::swap(tri.x1, tri.x2); std::swap(tri.y1, tri.y2); std::swap(tri.z1, tri.z2);
            tri.a01 = tri.y0 - tri.y1; tri.b01 = tri.x1 - tri.x0; tri.c01 = tri.x0 * tri.y1 - tri.x1 * tri.y0;
            tri.a12 = tri.y1 - tri.y2; tri.b12 = tri.x2 - tri.x1; tri.c12 = tri.x1 * tri.y2 - tri.x2 * tri.y1;
            tri.a20 = tri.y2 - tri.y0; tri.b20 = tri.x0 - tri.x2; tri.c20 = tri.x2 * tri.y0 - tri.x0 * tri.y2;
            area = -area;
        } else if (area <= 0.0f) {
            continue;
        }

        tri.inv_area = 1.0f / area;
        tri.nx0 = in.nx0; tri.ny0 = in.ny0; tri.nz0 = in.nz0;
        tri.nx1 = in.nx1; tri.ny1 = in.ny1; tri.nz1 = in.nz1;
        tri.nx2 = in.nx2; tri.ny2 = in.ny2; tri.nz2 = in.nz2;
        tri.r = in.r; tri.g = in.g; tri.b = in.b;
        tri.roughness = in.roughness;
        tri.is_hair = in.is_hair;

        int min_tx = std::clamp(static_cast<int>(std::min({tri.x0, tri.x1, tri.x2}) / 16.0f), 0, static_cast<int>(m_tiles_x - 1));
        int max_tx = std::clamp(static_cast<int>(std::max({tri.x0, tri.x1, tri.x2}) / 16.0f), 0, static_cast<int>(m_tiles_x - 1));
        int min_ty = std::clamp(static_cast<int>(std::min({tri.y0, tri.y1, tri.y2}) / 16.0f), 0, static_cast<int>(m_tiles_y - 1));
        int max_ty = std::clamp(static_cast<int>(std::max({tri.y0, tri.y1, tri.y2}) / 16.0f), 0, static_cast<int>(m_tiles_y - 1));

        uint32_t tri_idx = static_cast<uint32_t>(visible_triangles++);
        if (tri_idx >= m_transformed_triangles.size()) break;
        m_transformed_triangles[tri_idx] = tri;

        for (int ty = min_ty; ty <= max_ty; ++ty) {
            for (int tx = min_tx; tx <= max_tx; ++tx) {
                auto& tile = m_tiles[ty * m_tiles_x + tx];
                if (tile.tri_count < 2048) {
                    tile.tri_indices[tile.tri_count++] = tri_idx;
                }
            }
        }
    }

    struct TileJobPayload {
        SoftwareRendererAVX512* renderer;
        uint32_t* dst_color;
    } payload{this, dst_color_540p};

    pool.parallel_for([](void* user_data, uint32_t thread_id, uint32_t num_threads) {
        auto* p = static_cast<TileJobPayload*>(user_data);
        auto* renderer = p->renderer;

        const size_t tiles_per_core = (renderer->m_total_tiles + num_threads - 1) / num_threads;
        const size_t t_start = thread_id * tiles_per_core;
        const size_t t_end   = std::min(t_start + tiles_per_core, renderer->m_total_tiles);

        for (size_t t = t_start; t < t_end; ++t) {
            auto& tile = renderer->m_tiles[t];
            const size_t tile_x = (t % renderer->m_tiles_x) * RasterTile16x16::TILE_SIZE;
            const size_t tile_y = (t / renderer->m_tiles_x) * RasterTile16x16::TILE_SIZE;

            avx512_rasterize_tile_scanlines(renderer->m_transformed_triangles.data(), tile, tile_x, tile_y);

            for (size_t row = 0; row < 16; ++row) {
                if (tile_y + row >= renderer->m_render_h) break;
                const size_t dst_offset = (tile_y + row) * renderer->m_render_w + tile_x;
                const size_t src_offset = row * 16;
                std::memcpy(p->dst_color + dst_offset, tile.rgba + src_offset, 16 * sizeof(uint32_t));
            }
        }
    }, &payload);
}

} // namespace avx512
