#include "system2_coprocessor_suite.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

// ================================================================================================
// SYSTEM 2 SUITE (K1 - K6) IMPLEMENTATIONS
// ================================================================================================

void System2CoprocessorSuite::FrustumCullAndSerializeIndirect_AVX512(
    const BoundingSpheresSoA& spheres, size_t count, const float planes[6][4],
    const uint32_t* idx_counts, const uint32_t* first_indices, const int32_t* vertex_offsets,
    uint8_t* dst_indirect_bar, uint32_t* total_draws_out)
{
    FrustumCullAndSerializeIndirect_AVX512(
        spheres, count, reinterpret_cast<const float*>(planes),
        idx_counts, first_indices, vertex_offsets,
        dst_indirect_bar, total_draws_out
    );
}

void System2CoprocessorSuite::FrustumCullAndSerializeIndirect_AVX512(
    const BoundingSpheresSoA& spheres, size_t count, const float* planes,
    const uint32_t* idx_counts, const uint32_t* first_indices, const int32_t* vertex_offsets,
    uint8_t* dst_indirect_bar, uint32_t* total_draws_out)
{
    uint32_t visible_count = 0;
    auto* dst_cmds = reinterpret_cast<VkDrawIndexedIndirectCommand*>(dst_indirect_bar);

    for (size_t i = 0; i < count; i += 16) {
        const __m512 cx = _mm512_loadu_ps(spheres.cx + i);
        const __m512 cy = _mm512_loadu_ps(spheres.cy + i);
        const __m512 cz = _mm512_loadu_ps(spheres.cz + i);
        const __m512 cr = _mm512_loadu_ps(spheres.cr + i);

        __mmask16 inside_mask = 0xFFFF;

        for (int p = 0; p < 6; ++p) {
            const __m512 px = _mm512_set1_ps(planes[p * 4 + 0]);
            const __m512 py = _mm512_set1_ps(planes[p * 4 + 1]);
            const __m512 pz = _mm512_set1_ps(planes[p * 4 + 2]);
            const __m512 pd = _mm512_set1_ps(planes[p * 4 + 3]);

            __m512 dist = _mm512_fmadd_ps(cz, pz, _mm512_fmadd_ps(cy, py, _mm512_fmadd_ps(cx, px, pd)));
            __mmask16 plane_test = _mm512_cmp_ps_mask(dist, _mm512_sub_ps(_mm512_setzero_ps(), cr), _CMP_GE_OQ);
            inside_mask &= plane_test;
        }

        while (inside_mask) {
            int bit = __builtin_ctz(inside_mask);
            size_t idx = i + bit;
            if (idx < count) {
                dst_cmds[visible_count++] = {
                    idx_counts[idx], 1, first_indices[idx], vertex_offsets[idx], 0
                };
            }
            inside_mask &= inside_mask - 1;
        }
    }
    *total_draws_out = visible_count;
}

void System2CoprocessorSuite::CoalesceDrawCommands_AVX512(
    const DrawCallDescriptor* draws, size_t count,
    uint32_t* batch_offsets, uint32_t* batch_counts, uint32_t* total_batches_out)
{
    if (count == 0) {
        *total_batches_out = 0;
        return;
    }

    uint32_t batches = 0;
    uint32_t curr_offset = 0;
    uint32_t curr_count = 1;

    for (size_t i = 1; i < count; ++i) {
        bool same_batch = (draws[i].pso_id == draws[i - 1].pso_id) &&
                          (draws[i].pipeline_id == draws[i - 1].pipeline_id) &&
                          (draws[i].material_id == draws[i - 1].material_id);
        if (same_batch) {
            curr_count++;
        } else {
            batch_offsets[batches] = curr_offset;
            batch_counts[batches]  = curr_count;
            batches++;
            curr_offset = static_cast<uint32_t>(i);
            curr_count = 1;
        }
    }
    batch_offsets[batches] = curr_offset;
    batch_counts[batches]  = curr_count;
    *total_batches_out = batches + 1;
}

void System2CoprocessorSuite::AcousticSubdomainStep_AVX512(
    float* p_next, const float* p_curr, const float* p_prev, const uint8_t* abs_mask,
    float c, float dt, float dx)
{
    const float alpha = (c * dt / dx) * (c * dt / dx);
    const __m512 v_alpha = _mm512_set1_ps(alpha);
    const __m512 v_two   = _mm512_set1_ps(2.0f);

    for (size_t i = 0; i < 49152; i += 16) {
        const __m512 curr = _mm512_loadu_ps(p_curr + i);
        const __m512 prev = _mm512_loadu_ps(p_prev + i);

        __m512 next = _mm512_fmsub_ps(v_two, curr, prev);
        next = _mm512_fmadd_ps(v_alpha, curr, next);

        _mm512_storeu_ps(p_next + i, next);
    }
}

void System2CoprocessorSuite::InjectCameraMatrix_AVX512(
    const float* base_vp, float pitch_rate, float yaw_rate, float dt_pred, float fov_scale,
    CameraMatrixPayload* target_slot)
{
    const __m512 v_fov = _mm512_set1_ps(fov_scale);
    const __m512 r0 = _mm512_mul_ps(_mm512_loadu_ps(base_vp), v_fov);
    _mm512_store_ps(target_slot->view_proj, r0);
}

void System2CoprocessorSuite::RelaxChromaticMesh_AVX512(MeshParticlesSoA& mesh, const ChromaticEdgeSet& edges) {
    for (size_t i = 0; i < edges.count; ++i) {
        uint32_t idx1 = edges.p1[i];
        uint32_t idx2 = edges.p2[i];
        float dx = mesh.px[idx2] - mesh.px[idx1];
        float dy = mesh.py[idx2] - mesh.py[idx1];
        float dz = mesh.pz[idx2] - mesh.pz[idx1];
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz + 1e-6f);
        float diff = (dist - edges.rest_length[i]) / (dist * (mesh.inv_m[idx1] + mesh.inv_m[idx2]));
        
        mesh.px[idx1] += dx * diff * mesh.inv_m[idx1];
        mesh.py[idx1] += dy * diff * mesh.inv_m[idx1];
        mesh.pz[idx1] += dz * diff * mesh.inv_m[idx1];

        mesh.px[idx2] -= dx * diff * mesh.inv_m[idx2];
        mesh.py[idx2] -= dy * diff * mesh.inv_m[idx2];
        mesh.pz[idx2] -= dz * diff * mesh.inv_m[idx2];
    }
}

void System2CoprocessorSuite::RelaxChromaticMesh_AVX512_MT(
    MeshParticlesSoA& mesh, const std::vector<ChromaticEdgeSet>& chromatic_sets, CorePinnedThreadPool& pool)
{
    for (const auto& edge_set : chromatic_sets) {
        pool.parallel_for(edge_set.count, [&](size_t idx, size_t) {
            uint32_t idx1 = edge_set.p1[idx];
            uint32_t idx2 = edge_set.p2[idx];
            float dx = mesh.px[idx2] - mesh.px[idx1];
            float dy = mesh.py[idx2] - mesh.py[idx1];
            float dz = mesh.pz[idx2] - mesh.pz[idx1];
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz + 1e-6f);
            float diff = (dist - edge_set.rest_length[idx]) / (dist * (mesh.inv_m[idx1] + mesh.inv_m[idx2]));

            mesh.px[idx1] += dx * diff * mesh.inv_m[idx1];
            mesh.py[idx1] += dy * diff * mesh.inv_m[idx1];
            mesh.pz[idx1] += dz * diff * mesh.inv_m[idx1];

            mesh.px[idx2] -= dx * diff * mesh.inv_m[idx2];
            mesh.py[idx2] -= dy * diff * mesh.inv_m[idx2];
            mesh.pz[idx2] -= dz * diff * mesh.inv_m[idx2];
        });
    }
}

void System2CoprocessorSuite::MorphologicalSDFUIComposition_AVX512(
    const uint8_t* stencil, const uint8_t* r, const uint8_t* g, const uint8_t* sub_coords,
    uint8_t* dst_rgba, size_t num_pixels)
{
    for (size_t i = 0; i < num_pixels; i += 64) {
        __m512i st = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(stencil + i));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_rgba + i * 4), st);
    }
}

void System2CoprocessorSuite::MorphologicalSDFUIComposition_AVX512_MT(
    const uint8_t* stencil, const uint8_t* sub_coords, uint8_t* dst_ui,
    int width, int height, CorePinnedThreadPool& pool)
{
    pool.parallel_for(static_cast<size_t>(height), [&](size_t y, size_t) {
        const size_t offset = y * width;
        MorphologicalSDFUIComposition_AVX512(
            stencil + offset, nullptr, nullptr, sub_coords + offset * 4,
            dst_ui + offset * 4, width
        );
    });
}

// ================================================================================================
// FRONTIER KERNELS (K8 - K11) IMPLEMENTATIONS
// ================================================================================================

namespace avx512 {

void avx512_sieve_lights_for_cluster_range(
    const ClusterAABBArray& __restrict clusters,
    const PointLightViewSpace* __restrict lights,
    size_t num_lights,
    ClusterLightBitmaskTable& __restrict output_table,
    size_t packet_start,
    size_t packet_end)
{
    num_lights = std::min(num_lights, static_cast<size_t>(ClusterLightBitmaskTable::MAX_LIGHTS));

    for (size_t p = packet_start; p < packet_end; ++p) {
        const __m512 c_min_x = _mm512_load_ps(clusters.min_x[p]);
        const __m512 c_min_y = _mm512_load_ps(clusters.min_y[p]);
        const __m512 c_min_z = _mm512_load_ps(clusters.min_z[p]);
        const __m512 c_max_x = _mm512_load_ps(clusters.max_x[p]);
        const __m512 c_max_y = _mm512_load_ps(clusters.max_y[p]);
        const __m512 c_max_z = _mm512_load_ps(clusters.max_z[p]);
        const __m512 v_zero  = _mm512_setzero_ps();

        uint64_t accumulated_masks[16][4] = {0};

        for (size_t l = 0; l < num_lights; ++l) {
            const __m512 l_x = _mm512_set1_ps(lights[l].x);
            const __m512 l_y = _mm512_set1_ps(lights[l].y);
            const __m512 l_z = _mm512_set1_ps(lights[l].z);
            const __m512 l_r2 = _mm512_set1_ps(lights[l].radius * lights[l].radius);

            const __m512 dx = _mm512_add_ps(_mm512_max_ps(_mm512_sub_ps(c_min_x, l_x), v_zero),
                                            _mm512_max_ps(_mm512_sub_ps(l_x, c_max_x), v_zero));
            const __m512 dy = _mm512_add_ps(_mm512_max_ps(_mm512_sub_ps(c_min_y, l_y), v_zero),
                                            _mm512_max_ps(_mm512_sub_ps(l_y, c_max_y), v_zero));
            const __m512 dz = _mm512_add_ps(_mm512_max_ps(_mm512_sub_ps(c_min_z, l_z), v_zero),
                                            _mm512_max_ps(_mm512_sub_ps(l_z, c_max_z), v_zero));

            __m512 dist_sq = _mm512_mul_ps(dx, dx);
            dist_sq = _mm512_fmadd_ps(dy, dy, dist_sq);
            dist_sq = _mm512_fmadd_ps(dz, dz, dist_sq);

            const __mmask16 hit_mask = _mm512_cmple_ps_mask(dist_sq, l_r2);
            if (hit_mask != 0) {
                const size_t word_idx = l >> 6;
                const uint64_t bit_val = 1ULL << (l & 63);
                #pragma GCC unroll 16
                for (int i = 0; i < 16; ++i) {
                    if ((hit_mask >> i) & 1) accumulated_masks[i][word_idx] |= bit_val;
                }
            }
        }

        const size_t base_cluster = p * 16;
        for (int i = 0; i < 16; ++i) {
            output_table.light_masks[base_cluster + i][0] = accumulated_masks[i][0];
            output_table.light_masks[base_cluster + i][1] = accumulated_masks[i][1];
            output_table.light_masks[base_cluster + i][2] = accumulated_masks[i][2];
            output_table.light_masks[base_cluster + i][3] = accumulated_masks[i][3];
        }
    }
}

void avx512_wigner_ggx_compensate_row(
    const float* __restrict n_prev_x, const float* __restrict n_prev_y, const float* __restrict n_prev_z,
    const float* __restrict n_curr_x, const float* __restrict n_curr_y, const float* __restrict n_curr_z,
    const float* __restrict n_next_x, const float* __restrict n_next_y, const float* __restrict n_next_z,
    const float* __restrict roughness_in,
    float* __restrict roughness_out,
    size_t width)
{
    const __m512 v_kappa   = _mm512_set1_ps(2.0f);
    const __m512 v_one     = _mm512_set1_ps(1.0f);
    const __m512 v_quarter = _mm512_set1_ps(0.25f);

    for (size_t x = 0; x < width; x += 16) {
        const __m512 v_dy_x = _mm512_sub_ps(_mm512_loadu_ps(n_next_x + x), _mm512_loadu_ps(n_prev_x + x));
        const __m512 v_dy_y = _mm512_sub_ps(_mm512_loadu_ps(n_next_y + x), _mm512_loadu_ps(n_prev_x + x));
        const __m512 v_dy_z = _mm512_sub_ps(_mm512_loadu_ps(n_next_z + x), _mm512_loadu_ps(n_prev_x + x));
        __m512 v_dy_sq = _mm512_fmadd_ps(v_dy_z, v_dy_z, _mm512_fmadd_ps(v_dy_y, v_dy_y, _mm512_mul_ps(v_dy_x, v_dy_x)));

        const __m512 v_dx_x = _mm512_sub_ps(_mm512_loadu_ps(n_curr_x + x + 1), _mm512_loadu_ps(n_curr_x + x - 1));
        const __m512 v_dx_y = _mm512_sub_ps(_mm512_loadu_ps(n_curr_y + x + 1), _mm512_loadu_ps(n_curr_x + x - 1));
        const __m512 v_dx_z = _mm512_sub_ps(_mm512_loadu_ps(n_curr_z + x + 1), _mm512_loadu_ps(n_curr_z + x - 1));
        __m512 v_dx_sq = _mm512_fmadd_ps(v_dx_z, v_dx_z, _mm512_fmadd_ps(v_dx_y, v_dx_y, _mm512_mul_ps(v_dx_x, v_dx_x)));

        const __m512 v_var_n = _mm512_mul_ps(_mm512_add_ps(v_dx_sq, v_dy_sq), v_quarter);

        const __m512 v_alpha_base = _mm512_loadu_ps(roughness_in + x);
        const __m512 v_alpha_sq   = _mm512_mul_ps(v_alpha_base, v_alpha_base);
        const __m512 v_alpha_eff2 = _mm512_min_ps(_mm512_fmadd_ps(v_kappa, v_var_n, v_alpha_sq), v_one);

        _mm512_storeu_ps(roughness_out + x, _mm512_sqrt_ps(v_alpha_eff2));
    }
}

__m512 avx512_aces_fitted_ps(__m512 x) {
    const __m512 a = _mm512_set1_ps(2.51f);
    const __m512 b = _mm512_set1_ps(0.03f);
    const __m512 c = _mm512_set1_ps(2.43f);
    const __m512 d = _mm512_set1_ps(0.59f);
    const __m512 e = _mm512_set1_ps(0.14f);
    const __m512 v_two  = _mm512_set1_ps(2.0f);
    const __m512 v_zero = _mm512_setzero_ps();
    const __m512 v_one  = _mm512_set1_ps(1.0f);

    x = _mm512_max_ps(x, v_zero);
    const __m512 num = _mm512_mul_ps(x, _mm512_fmadd_ps(a, x, b));
    const __m512 den = _mm512_fmadd_ps(x, _mm512_fmadd_ps(c, x, d), e);

    const __m512 rcp_approx = _mm512_rcp14_ps(den);
    const __m512 rcp_refined = _mm512_mul_ps(rcp_approx, _mm512_fnmadd_ps(den, rcp_approx, v_two));

    return _mm512_min_ps(_mm512_mul_ps(num, rcp_refined), v_one);
}

void avx512_tonemap_row(
    const float* __restrict r_in, const float* __restrict g_in, const float* __restrict b_in,
    uint32_t* __restrict ldr_out, size_t width)
{
    const __m512 v_255 = _mm512_set1_ps(255.0f);
    for (size_t x = 0; x < width; x += 16) {
        const __m512 r = avx512_aces_fitted_ps(_mm512_loadu_ps(r_in + x));
        const __m512 g = avx512_aces_fitted_ps(_mm512_loadu_ps(g_in + x));
        const __m512 b = avx512_aces_fitted_ps(_mm512_loadu_ps(b_in + x));

        const __m512i r_u32 = _mm512_cvttps_epi32(_mm512_mul_ps(r, v_255));
        const __m512i g_u32 = _mm512_cvttps_epi32(_mm512_mul_ps(g, v_255));
        const __m512i b_u32 = _mm512_cvttps_epi32(_mm512_mul_ps(b, v_255));
        const __m512i a_u32 = _mm512_set1_epi32(255);

        const __m512i rgba = _mm512_or_epi32(
            _mm512_or_epi32(r_u32, _mm512_slli_epi32(g_u32, 8)),
            _mm512_or_epi32(_mm512_slli_epi32(b_u32, 16), _mm512_slli_epi32(a_u32, 24))
        );
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ldr_out + x), rgba);
    }
}

CoprocessorEngine::CoprocessorEngine() {
    m_probe_grid.resize(256);
    initialize_clusters(1.0472f, 16.0f / 9.0f, 0.1f, 100.0f);
}

void CoprocessorEngine::initialize_clusters(float fov_y, float aspect, float z_near, float z_far) {
    const float tan_half_fov = std::tan(fov_y * 0.5f);
    const float log_ratio = std::log(z_far / z_near);

    for (size_t kz = 0; kz < 32; ++kz) {
        const float z_k0 = z_near * std::exp(log_ratio * (float(kz) / 32.0f));
        const float z_k1 = z_near * std::exp(log_ratio * (float(kz + 1) / 32.0f));

        for (size_t ky = 0; ky < 16; ++ky) {
            for (size_t kx = 0; kx < 16; ++kx) {
                const size_t cluster_idx = kz * 256 + ky * 16 + kx;
                const size_t packet_idx  = cluster_idx / 16;
                const size_t lane_idx    = cluster_idx % 16;

                const float x0_ndc = -1.0f + float(kx) * (2.0f / 16.0f);
                const float x1_ndc = -1.0f + float(kx + 1) * (2.0f / 16.0f);
                const float y0_ndc = -1.0f + float(ky) * (2.0f / 16.0f);
                const float y1_ndc = -1.0f + float(ky + 1) * (2.0f / 16.0f);

                m_clusters.min_x[packet_idx][lane_idx] = x0_ndc * z_k1 * tan_half_fov * aspect;
                m_clusters.max_x[packet_idx][lane_idx] = x1_ndc * z_k1 * tan_half_fov * aspect;
                m_clusters.min_y[packet_idx][lane_idx] = y0_ndc * z_k1 * tan_half_fov;
                m_clusters.max_y[packet_idx][lane_idx] = y1_ndc * z_k1 * tan_half_fov;
                m_clusters.min_z[packet_idx][lane_idx] = z_k0;
                m_clusters.max_z[packet_idx][lane_idx] = z_k1;
            }
        }
    }
}

struct FrameJobPayload {
    CoprocessorEngine* engine;
    const PointLightViewSpace* lights;
    size_t num_lights;
    const HDRPixelBuffer* hdr_in;
    uint32_t* ldr_out;
};

void CoprocessorEngine::execute_frame(
    const PointLightViewSpace* lights, size_t num_lights,
    const float*, float*,
    const HDRPixelBuffer& hdr_in, uint32_t* ldr_out,
    size_t, size_t)
{
    FrameJobPayload payload{this, lights, num_lights, &hdr_in, ldr_out};

    m_pool.parallel_for([](void* user_data, uint32_t thread_id, uint32_t num_threads) {
        auto* p = static_cast<FrameJobPayload*>(user_data);

        // 1. K8: Cluster Sieve
        const size_t total_packets = ClusterAABBArray::NUM_PACKETS;
        const size_t packets_per_thread = (total_packets + num_threads - 1) / num_threads;
        const size_t p_start = thread_id * packets_per_thread;
        const size_t p_end   = std::min(p_start + packets_per_thread, total_packets);

        if (p_start < p_end) {
            avx512_sieve_lights_for_cluster_range(
                p->engine->m_clusters, p->lights, p->num_lights,
                p->engine->m_light_table, p_start, p_end);
        }

        // 2. K11: Tonemapping
        const size_t total_rows = p->hdr_in->height;
        const size_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
        const size_t r_start = thread_id * rows_per_thread;
        const size_t r_end   = std::min(r_start + rows_per_thread, total_rows);

        for (size_t y = r_start; y < r_end; ++y) {
            const size_t row_offset = y * p->hdr_in->width;
            avx512_tonemap_row(
                p->hdr_in->r.data() + row_offset,
                p->hdr_in->g.data() + row_offset,
                p->hdr_in->b.data() + row_offset,
                p->ldr_out + row_offset,
                p->hdr_in->width
            );
        }
    }, &payload);
}

} // namespace avx512
