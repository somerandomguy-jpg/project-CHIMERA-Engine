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
#include <cstdint>
#include <chrono>

struct DrsSettings {
    double target_frametime_ms{16.666};
    double min_scale{0.50};
    double max_scale{1.00};
    double step_up_cooldown_ms{600.0};
    double ema_alpha{0.15};
    bool enabled{false};
};

class DrsController {
public:
    DrsController(int native_w, int native_h, DrsSettings settings = {});
    void update(double frame_time_ms);
    int get_render_width() const { return current_w_; }
    int get_render_height() const { return current_h_; }
    double get_current_scale() const { return current_scale_; }
    bool is_enabled() const { return settings_.enabled; }
    void toggle() { settings_.enabled = !settings_.enabled; }

private:
    int native_w_;
    int native_h_;
    DrsSettings settings_;
    double current_scale_{1.0};
    int current_w_;
    int current_h_;
    double smoothed_frametime_ms_{0.0};
    std::chrono::high_resolution_clock::time_point last_scale_change_time_;
};
