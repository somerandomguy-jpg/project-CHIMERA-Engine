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

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <cstdint>
#include <string>

class X11WindowCapture {
public:
    explicit X11WindowCapture(unsigned long target_window = 0);
    ~X11WindowCapture();

    bool init_shm(int w, int h);
    void cleanup_shm();

    bool capture_frame(uint32_t* dst, int max_w, int max_h, int& out_w, int& out_h);
    void forward_key(int keycode, bool press);
    void forward_mouse_click_atomic(double x, double y, int btn, int win_w, int win_h, bool press);

    bool find_target_window();
    std::string get_status_str() const;

private:
    Display* display_{nullptr};
    Window target_window_{0};
    XImage* ximage_{nullptr};
    XShmSegmentInfo shminfo_{};
    bool shm_attached_{false};
    int width_{0};
    int height_{0};
    std::string found_title_;
};
