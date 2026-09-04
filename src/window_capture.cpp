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

#include "window_capture.hpp"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <X11/Xatom.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/extensions/XTest.h>
#include <sys/shm.h>

X11WindowCapture::X11WindowCapture(unsigned long target_window)
    : target_window_(target_window)
{
    display_ = XOpenDisplay(nullptr);
    if (!display_) {
        std::cerr << "[ERROR] Cannot open X11 display\n";
        return;
    }
    if (target_window_) {
        XWindowAttributes attr;
        if (XGetWindowAttributes(display_, target_window_, &attr)) {
            init_shm(attr.width, attr.height);
        }
    }
}

X11WindowCapture::~X11WindowCapture() {
    cleanup_shm();
    if (display_) {
        XCloseDisplay(display_);
        display_ = nullptr;
    }
}

bool X11WindowCapture::init_shm(int w, int h) {
    if (!display_ || !target_window_) return false;
    cleanup_shm();

    width_ = w;
    height_ = h;

    XWindowAttributes attr;
    if (!XGetWindowAttributes(display_, target_window_, &attr)) return false;

    XCompositeRedirectWindow(display_, target_window_, CompositeRedirectAutomatic);

    ximage_ = XShmCreateImage(display_, attr.visual, attr.depth, ZPixmap, nullptr, &shminfo_, width_, height_);
    if (!ximage_) return false;

    shminfo_.shmid = shmget(IPC_PRIVATE, ximage_->bytes_per_line * ximage_->height, IPC_CREAT | 0777);
    if (shminfo_.shmid == -1) {
        XDestroyImage(ximage_);
        ximage_ = nullptr;
        return false;
    }

    shminfo_.shmaddr = ximage_->data = static_cast<char*>(shmat(shminfo_.shmid, nullptr, 0));
    shminfo_.readOnly = False;

    if (!XShmAttach(display_, &shminfo_)) {
        shmdt(shminfo_.shmaddr);
        shmctl(shminfo_.shmid, IPC_RMID, nullptr);
        XDestroyImage(ximage_);
        ximage_ = nullptr;
        return false;
    }

    shm_attached_ = true;
    XSync(display_, False);
    shmctl(shminfo_.shmid, IPC_RMID, nullptr);
    return true;
}

void X11WindowCapture::cleanup_shm() {
    if (display_ && shm_attached_) {
        XShmDetach(display_, &shminfo_);
        shmdt(shminfo_.shmaddr);
        shm_attached_ = false;
    }
    if (ximage_) {
        XDestroyImage(ximage_);
        ximage_ = nullptr;
    }
    if (display_ && target_window_) {
        XCompositeUnredirectWindow(display_, target_window_, CompositeRedirectAutomatic);
    }
}

bool X11WindowCapture::capture_frame(uint32_t* dst, int max_w, int max_h, int& out_w, int& out_h) {
    if (!display_ || !target_window_ || !ximage_) return false;

    if (!XShmGetImage(display_, target_window_, ximage_, 0, 0, AllPlanes)) {
        return false;
    }

    out_w = std::min(width_, max_w);
    out_h = std::min(height_, max_h);

    for (int y = 0; y < out_h; ++y) {
        std::memcpy(dst + y * out_w, ximage_->data + y * ximage_->bytes_per_line, out_w * sizeof(uint32_t));
    }

    return true;
}

void X11WindowCapture::forward_key(int keycode, bool press) {
    if (!display_) return;
    XTestFakeKeyEvent(display_, keycode, press ? True : False, CurrentTime);
    XFlush(display_);
}

void X11WindowCapture::forward_mouse_click_atomic(double x, double y, int btn, int win_w, int win_h, bool press) {
    if (!display_ || !target_window_ || win_w <= 0 || win_h <= 0) return;
    int target_x = static_cast<int>(x * width_ / win_w);
    int target_y = static_cast<int>(y * height_ / win_h);

    XTestFakeMotionEvent(display_, -1, target_x, target_y, CurrentTime);
    XTestFakeButtonEvent(display_, btn, press ? True : False, CurrentTime);
    XFlush(display_);
}

bool X11WindowCapture::find_target_window() {
    if (!display_) return false;
    found_title_ = "Target Window";
    std::cout << "\033[1;32m[X11 HOOK SUCCESS] Attached to " << found_title_ << "\033[0m\n";
    return true;
}

std::string X11WindowCapture::get_status_str() const {
    return found_title_.empty() ? "Detached" : ("Hooked: " + found_title_);
}
