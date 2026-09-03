#pragma once
#ifndef WINDOW_CAPTURE_HPP
#define WINDOW_CAPTURE_HPP

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/extensions/XShm.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/extensions/XTest.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string>
#include <cstdint>
#include <chrono>

class X11WindowCapture {
public:
    X11WindowCapture(Window ignore_window = 0);
    ~X11WindowCapture();

    void set_ignore_window(Window w) { ignore_window_ = w; }
    bool find_target_window();
    bool capture_frame(uint32_t* dst_buffer, int max_w, int max_h, int& out_w, int& out_h);
    bool is_ready() const { return target_window_ != 0 && shm_ready_; }
    std::string get_status_str() const;

    void focus_tf2();
    void forward_mouse_click_atomic(double x_1080, double y_1080, int native_w, int native_h, int button, bool is_press);
    void forward_key(int scancode, bool is_press);

    // Mouse Delta Tracking for Asynchronous Frame Extrapolation
    void get_accumulated_mouse_delta(float& out_dx, float& out_dy) {
        out_dx = accumulated_mouse_dx_;
        out_dy = accumulated_mouse_dy_;
        accumulated_mouse_dx_ = 0.0f;
        accumulated_mouse_dy_ = 0.0f;
    }
    void inject_mouse_motion(float dx, float dy) {
        accumulated_mouse_dx_ += dx;
        accumulated_mouse_dy_ += dy;
    }

private:
    void cleanup_shm();
    bool init_shm(int w, int h);
    bool is_target_tf2_window(Window win);
    void update_geometry_cache();

    Display* display_{nullptr};
    Window root_{0};
    Window ignore_window_{0};
    Window target_window_{0};
    std::string found_title_;
    XImage* ximage_{nullptr};
    XShmSegmentInfo shminfo_{};
    bool shm_ready_{false};
    int current_w_{0};
    int current_h_{0};
    int win_origin_x_{0};
    int win_origin_y_{0};

    // Geometry Caching (Eliminates synchronous X server stalls)
    std::chrono::steady_clock::time_point last_geom_time_;
    std::chrono::steady_clock::time_point last_search_time_;

    // Asynchronous mouse delta accumulation
    float accumulated_mouse_dx_{0.0f};
    float accumulated_mouse_dy_{0.0f};
};
#endif
