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
