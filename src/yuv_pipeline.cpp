#include "yuv_pipeline.hpp"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

void Yuv420Frame::resize(int w, int h) {
    width = w;
    height = h;
    y_stride = (w + 63) & ~63;
    uv_stride = ((w / 2) + 63) & ~63;
    y_plane.resize(y_stride * h);
    u_plane.resize(uv_stride * (h / 2));
    v_plane.resize(uv_stride * (h / 2));
}

static inline __m512i pack_rgba_from_ps(__m512 r, __m512 g, __m512 b, __m512 a) {
    __m512 zero = _mm512_setzero_ps();
    __m512 max_v = _mm512_set1_ps(255.0f);

    __m512i ri = _mm512_cvttps_epi32(_mm512_min_ps(_mm512_max_ps(r, zero), max_v));
    __m512i gi = _mm512_cvttps_epi32(_mm512_min_ps(_mm512_max_ps(g, zero), max_v));
    __m512i bi = _mm512_cvttps_epi32(_mm512_min_ps(_mm512_max_ps(b, zero), max_v));
    __m512i ai = _mm512_cvttps_epi32(_mm512_min_ps(_mm512_max_ps(a, zero), max_v));

    return _mm512_or_si512(
        _mm512_or_si512(_mm512_slli_epi32(ai, 24), _mm512_slli_epi32(ri, 16)),
        _mm512_or_si512(_mm512_slli_epi32(gi, 8), bi)
    );
}

void YuvSimdConverter::convert_yuv420p_to_rgba(const Yuv420Frame& frame,
                                              uint32_t* dst_rgba,
                                              int dst_stride,
                                              CorePinnedThreadPool& pool) {
    const int width = frame.width;
    const int height = frame.height;
    const int y_stride = frame.y_stride;
    const int uv_stride = frame.uv_stride;
    const uint8_t* y_base = frame.y_plane.data();
    const uint8_t* u_base = frame.u_plane.data();
    const uint8_t* v_base = frame.v_plane.data();

    const __m512 c_r_v = _mm512_set1_ps(1.402f);
    const __m512 c_g_u = _mm512_set1_ps(-0.344136f);
    const __m512 c_g_v = _mm512_set1_ps(-0.714136f);
    const __m512 c_b_u = _mm512_set1_ps(1.772f);
    const __m512 alpha = _mm512_set1_ps(255.0f);
    const __m512i c_128 = _mm512_set1_epi32(128);

    pool.parallel_for(height, [=](size_t y_idx, size_t) {
        int y = static_cast<int>(y_idx);
        const uint8_t* y_row = y_base + y * y_stride;
        const uint8_t* u_row = u_base + (y / 2) * uv_stride;
        const uint8_t* v_row = v_base + (y / 2) * uv_stride;
        uint32_t* dst_row = dst_rgba + y * dst_stride;

        int x = 0;
        for (; x + 32 <= width; x += 32) {
            __m128i y_raw_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y_row + x));
            uint64_t u_val_0 = *reinterpret_cast<const uint64_t*>(u_row + x / 2);
            uint64_t v_val_0 = *reinterpret_cast<const uint64_t*>(v_row + x / 2);
            __m128i u_dup_0 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(u_val_0), _mm_cvtsi64_si128(u_val_0));
            __m128i v_dup_0 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(v_val_0), _mm_cvtsi64_si128(v_val_0));

            __m512 yf_0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(y_raw_0));
            __m512 uf_0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(u_dup_0), c_128));
            __m512 vf_0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(v_dup_0), c_128));

            __m512 r_0 = _mm512_fmadd_ps(vf_0, c_r_v, yf_0);
            __m512 g_0 = _mm512_fmadd_ps(uf_0, c_g_u, _mm512_fmadd_ps(vf_0, c_g_v, yf_0));
            __m512 b_0 = _mm512_fmadd_ps(uf_0, c_b_u, yf_0);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row + x), pack_rgba_from_ps(r_0, g_0, b_0, alpha));

            __m128i y_raw_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y_row + x + 16));
            uint64_t u_val_1 = *reinterpret_cast<const uint64_t*>(u_row + (x + 16) / 2);
            uint64_t v_val_1 = *reinterpret_cast<const uint64_t*>(v_row + (x + 16) / 2);
            __m128i u_dup_1 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(u_val_1), _mm_cvtsi64_si128(u_val_1));
            __m128i v_dup_1 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(v_val_1), _mm_cvtsi64_si128(v_val_1));

            __m512 yf_1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(y_raw_1));
            __m512 uf_1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(u_dup_1), c_128));
            __m512 vf_1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(v_dup_1), c_128));

            __m512 r_1 = _mm512_fmadd_ps(vf_1, c_r_v, yf_1);
            __m512 g_1 = _mm512_fmadd_ps(uf_1, c_g_u, _mm512_fmadd_ps(vf_1, c_g_v, yf_1));
            __m512 b_1 = _mm512_fmadd_ps(uf_1, c_b_u, yf_1);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_row + x + 16), pack_rgba_from_ps(r_1, g_1, b_1, alpha));
        }

        for (; x < width; ++x) {
            float yv = static_cast<float>(y_row[x]);
            float uv = static_cast<float>(u_row[x / 2]) - 128.0f;
            float vv = static_cast<float>(v_row[x / 2]) - 128.0f;
            uint8_t r = static_cast<uint8_t>(std::clamp(yv + 1.402f * vv, 0.0f, 255.0f));
            uint8_t g = static_cast<uint8_t>(std::clamp(yv - 0.344136f * uv - 0.714136f * vv, 0.0f, 255.0f));
            uint8_t b = static_cast<uint8_t>(std::clamp(yv + 1.772f * uv, 0.0f, 255.0f));
            dst_row[x] = (255u << 24) | (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(b);
        }
    });
    _mm_sfence();
}

VideoIngestionPipeline::VideoIngestionPipeline(int width, int height)
    : width_(width), height_(height) {
    current_frame_.resize(width, height);
}

void VideoIngestionPipeline::generate_synthetic_frame(float timestamp) {
    uint8_t* y_base = current_frame_.y_plane.data();
    uint8_t* u_base = current_frame_.u_plane.data();
    uint8_t* v_base = current_frame_.v_plane.data();
    const int w = width_;
    const int h = height_;
    const int y_stride = current_frame_.y_stride;
    const int uv_stride = current_frame_.uv_stride;

    float center_x = w * 0.5f + std::sin(timestamp * 1.5f) * (w * 0.25f);
    float center_y = h * 0.5f + std::cos(timestamp * 1.5f) * (h * 0.25f);

    for (int y = 0; y < h; ++y) {
        uint8_t* y_row = y_base + y * y_stride;
        float dy = y - center_y;
        for (int x = 0; x < w; ++x) {
            float dx = x - center_x;
            float dist = std::sqrt(dx * dx + dy * dy);
            float angle = std::atan2(dy, dx);

            float ring_val = std::sin(dist * 0.25f - timestamp * 4.0f) * 0.5f + 0.5f;
            float star_val = std::sin(angle * 16.0f + timestamp * 2.0f) * 0.5f + 0.5f;
            float grid_val = ((x % 24 == 0) || (y % 24 == 0)) ? 1.0f : 0.0f;

            float intensity = 0.5f * ring_val + 0.3f * star_val + 0.2f * grid_val;
            y_row[x] = static_cast<uint8_t>(std::clamp(intensity * 235.0f + 16.0f, 16.0f, 235.0f));
        }
    }

    for (int cy = 0; cy < h / 2; ++cy) {
        uint8_t* u_row = u_base + cy * uv_stride;
        uint8_t* v_row = v_base + cy * uv_stride;
        for (int cx = 0; cx < w / 2; ++cx) {
            float u_val = std::sin(cx * 0.04f + timestamp) * 80.0f + 128.0f;
            float v_val = std::cos(cy * 0.04f + timestamp) * 80.0f + 128.0f;
            u_row[cx] = static_cast<uint8_t>(std::clamp(u_val, 16.0f, 240.0f));
            v_row[cx] = static_cast<uint8_t>(std::clamp(v_val, 16.0f, 240.0f));
        }
    }
}
