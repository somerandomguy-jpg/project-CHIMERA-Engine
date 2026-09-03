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
