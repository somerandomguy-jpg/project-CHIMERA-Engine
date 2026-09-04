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

#include "drs_controller.hpp"
#include <algorithm>
#include <cmath>

DrsController::DrsController(int native_w, int native_h, DrsSettings settings)
    : native_w_(native_w),
      native_h_(native_h),
      settings_(settings),
      current_scale_(settings.max_scale),
      current_w_(native_w),
      current_h_(native_h),
      smoothed_frametime_ms_(settings.target_frametime_ms),
      last_scale_change_time_(std::chrono::high_resolution_clock::now()) {}

void DrsController::update(double frame_time_ms) {
    if (smoothed_frametime_ms_ <= 0.0) {
        smoothed_frametime_ms_ = frame_time_ms;
    } else {
        smoothed_frametime_ms_ = (1.0 - settings_.ema_alpha) * smoothed_frametime_ms_ + settings_.ema_alpha * frame_time_ms;
    }

    if (!settings_.enabled) {
        current_scale_ = settings_.max_scale;
        current_w_ = native_w_;
        current_h_ = native_h_;
        return;
    }

    auto now = std::chrono::high_resolution_clock::now();
    double time_since_change_ms = std::chrono::duration<double, std::milli>(now - last_scale_change_time_).count();

    if (smoothed_frametime_ms_ > settings_.target_frametime_ms * 0.92) {
        double new_scale = std::max(settings_.min_scale, current_scale_ - 0.05);
        if (std::abs(new_scale - current_scale_) > 0.001) {
            current_scale_ = new_scale;
            last_scale_change_time_ = now;
        }
    } else if (smoothed_frametime_ms_ < settings_.target_frametime_ms * 0.75 &&
               time_since_change_ms >= settings_.step_up_cooldown_ms) {
        double new_scale = std::min(settings_.max_scale, current_scale_ + 0.05);
        if (std::abs(new_scale - current_scale_) > 0.001) {
            current_scale_ = new_scale;
            last_scale_change_time_ = now;
        }
    }

    current_w_ = (static_cast<int>(native_w_ * current_scale_) + 15) & ~15;
    current_h_ = (static_cast<int>(native_h_ * current_scale_) + 1) & ~1;
}
