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
#ifndef YUV_PIPELINE_HPP
#define YUV_PIPELINE_HPP

#include <cstdint>
#include <vector>
#include "thread_pool.hpp"
#include "aligned_buffer.hpp"

struct Yuv420Frame {
    int width{0};
    int height{0};
    int y_stride{0};
    int uv_stride{0};
    AlignedBuffer<uint8_t, 64> y_plane;
    AlignedBuffer<uint8_t, 64> u_plane;
    AlignedBuffer<uint8_t, 64> v_plane;

    void resize(int w, int h);
};

class YuvSimdConverter {
public:
    static void convert_yuv420p_to_rgba(const Yuv420Frame& frame,
                                        uint32_t* dst_rgba,
                                        int dst_stride,
                                        CorePinnedThreadPool& pool);
};

class VideoIngestionPipeline {
public:
    VideoIngestionPipeline(int width, int height);
    void generate_synthetic_frame(float timestamp);
    const Yuv420Frame& get_current_frame() const { return current_frame_; }

private:
    int width_;
    int height_;
    Yuv420Frame current_frame_;
};

#endif // YUV_PIPELINE_HPP
