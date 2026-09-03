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
