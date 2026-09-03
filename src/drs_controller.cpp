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
