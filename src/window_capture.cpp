#include "window_capture.hpp"
#include <X11/Xatom.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/extensions/XTest.h>
#include <iostream>
#include <cstring>
#include <algorithm>

static int x11_silent_error_handler(Display*, XErrorEvent*) {
    return 0; // Prevent X11 from crashing our engine on window resize/minimize
}

X11WindowCapture::X11WindowCapture(Window ignore_window)
    : ignore_window_(ignore_window) {
    XInitThreads();
    XSetErrorHandler(x11_silent_error_handler);
    display_ = XOpenDisplay(nullptr);
    if (display_) {
        root_ = DefaultRootWindow(display_);
    }
    last_geom_time_ = std::chrono::steady_clock::now();
    last_search_time_ = std::chrono::steady_clock::now();
}

X11WindowCapture::~X11WindowCapture() {
    cleanup_shm();
    if (display_) {
        XCloseDisplay(display_);
        display_ = nullptr;
    }
}

void X11WindowCapture::cleanup_shm() {
    if (display_) {
        if (target_window_) {
            XCompositeUnredirectWindow(display_, target_window_, CompositeRedirectAutomatic);
        }
        if (shm_ready_) {
            XShmDetach(display_, &shminfo_);
            XDestroyImage(ximage_);
            shmdt(shminfo_.shmaddr);
            shmctl(shminfo_.shmid, IPC_RMID, nullptr);
            shm_ready_ = false;
            ximage_ = nullptr;
        }
    }
}

void X11WindowCapture::focus_tf2() {
    if (!display_ || target_window_ == 0) return;
    XSetInputFocus(display_, target_window_, RevertToPointerRoot, CurrentTime);
    XFlush(display_);
}

bool X11WindowCapture::init_shm(int w, int h) {
    if (!display_ || !target_window_ || w <= 0 || h <= 0) return false;

    XWindowAttributes attr;
    if (!XGetWindowAttributes(display_, target_window_, &attr)) {
        return false;
    }

    ximage_ = XShmCreateImage(display_, attr.visual, attr.depth,
                              ZPixmap, nullptr, &shminfo_, w, h);
    if (!ximage_) return false;

    shminfo_.shmid = shmget(IPC_PRIVATE, ximage_->bytes_per_line * ximage_->height, IPC_CREAT | 0777);
    if (shminfo_.shmid < 0) {
        XDestroyImage(ximage_);
        ximage_ = nullptr;
        return false;
    }

    shminfo_.shmaddr = ximage_->data = static_cast<char*>(shmat(shminfo_.shmid, nullptr, 0));
    shminfo_.readOnly = False;

    if (!XShmAttach(display_, &shminfo_)) {
        cleanup_shm();
        return false;
    }

    // Direct off-screen redirection: captures only TF2 without locking multi-monitor desktop
    XCompositeRedirectWindow(display_, target_window_, CompositeRedirectAutomatic);
    XSync(display_, False);

    current_w_ = w;
    current_h_ = h;
    shm_ready_ = true;
    return true;
}

static std::string get_window_name_utf8(Display* display, Window win) {
    Atom net_wm_name = XInternAtom(display, "_NET_WM_NAME", True);
    Atom utf8_string = XInternAtom(display, "UTF8_STRING", True);
    if (net_wm_name != None && utf8_string != None) {
        Atom type; int format; unsigned long nitems, bytes_after;
        unsigned char* prop = nullptr;
        if (XGetWindowProperty(display, win, net_wm_name, 0, 1024, False, utf8_string,
                               &type, &format, &nitems, &bytes_after, &prop) == Success && prop) {
            std::string result(reinterpret_cast<char*>(prop));
            XFree(prop);
            return result;
        }
    }
    char* name = nullptr;
    if (XFetchName(display, win, &name) > 0 && name) {
        std::string result(name);
        XFree(name);
        return result;
    }
    return "";
}

bool X11WindowCapture::is_target_tf2_window(Window win) {
    if (win == 0 || win == root_ || win == ignore_window_) return false;

    std::string win_name = get_window_name_utf8(display_, win);
    if (!win_name.empty()) {
        if (win_name.find("Presentation") != std::string::npos || win_name.find("Super-Resolution") != std::string::npos) {
            return false;
        }
        if (win_name.find("Team Fortress 2") != std::string::npos || win_name.find("Team Fortress") != std::string::npos) {
            found_title_ = win_name;
            return true;
        }
    }

    XClassHint class_hint;
    if (XGetClassHint(display_, win, &class_hint)) {
        std::string res_name = class_hint.res_name ? class_hint.res_name : "";
        std::string res_class = class_hint.res_class ? class_hint.res_class : "";
        if (class_hint.res_name) XFree(class_hint.res_name);
        if (class_hint.res_class) XFree(class_hint.res_class);

        std::transform(res_name.begin(), res_name.end(), res_name.begin(), ::tolower);
        std::transform(res_class.begin(), res_class.end(), res_class.begin(), ::tolower);

        if (res_name.find("hl2_linux") != std::string::npos || res_name.find("tf_linux") != std::string::npos ||
            res_class.find("hl2_linux") != std::string::npos || res_class.find("tf_linux") != std::string::npos ||
            res_name.find("steam_app_440") != std::string::npos) {
            found_title_ = "Team Fortress 2 (" + res_name + ")";
            return true;
        }
    }
    return false;
}

bool X11WindowCapture::find_target_window() {
    if (!display_) return false;

    Atom net_client_list = XInternAtom(display_, "_NET_CLIENT_LIST", True);
    if (net_client_list != None) {
        Atom type; int format; unsigned long nitems, bytes_after;
        unsigned char* data = nullptr;
        if (XGetWindowProperty(display_, root_, net_client_list, 0, 1024, False, XA_WINDOW,
                               &type, &format, &nitems, &bytes_after, &data) == Success && data) {
            Window* windows = reinterpret_cast<Window*>(data);
            for (unsigned long i = 0; i < nitems; ++i) {
                if (is_target_tf2_window(windows[i])) {
                    target_window_ = windows[i];
                    XFree(data);
                    std::cout << "\n\033[1;32m[X11 HOOK SUCCESS] Attached to " << found_title_ << "\033[0m\n" << std::endl;
                    return true;
                }
            }
            XFree(data);
        }
    }
    return false;
}

void X11WindowCapture::update_geometry_cache() {
    auto now = std::chrono::steady_clock::now();
    // Cache geometry for 100ms (10 Hz) -> Eliminates 1000 FPS X11 socket roundtrips
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_geom_time_).count() > 100) {
        last_geom_time_ = now;
        Window root_ret; int x_ret, y_ret; unsigned int w_ret = 0, h_ret = 0, b_ret, d_ret;
        if (XGetGeometry(display_, target_window_, &root_ret, &x_ret, &y_ret, &w_ret, &h_ret, &b_ret, &d_ret)) {
            Window child_ret;
            XTranslateCoordinates(display_, target_window_, root_, 0, 0, &win_origin_x_, &win_origin_y_, &child_ret);
            current_w_ = static_cast<int>(w_ret);
            current_h_ = static_cast<int>(h_ret);
        }
    }
}

bool X11WindowCapture::capture_frame(uint32_t* dst_buffer, int max_w, int max_h, int& out_w, int& out_h) {
    if (!display_) return false;

    if (target_window_ == 0) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_search_time_).count() > 500) {
            last_search_time_ = now;
            find_target_window();
        }
        return false;
    }

    update_geometry_cache();
    if (current_w_ <= 0 || current_h_ <= 0) return false;

    int w = std::min(current_w_, max_w);
    int h = std::min(current_h_, max_h);

    if (!shm_ready_ || ximage_->width != current_w_ || ximage_->height != current_h_) {
        cleanup_shm();
        if (!init_shm(current_w_, current_h_)) return false;
    }

    // Direct XShm extraction from dedicated off-screen backing pixmap (0 lock on root)
    if (!XShmGetImage(display_, target_window_, ximage_, 0, 0, AllPlanes)) {
        return false;
    }

    int src_stride = ximage_->bytes_per_line / 4;
    const uint32_t* src_pixels = reinterpret_cast<const uint32_t*>(ximage_->data);

    for (int y = 0; y < h; ++y) {
        std::memcpy(dst_buffer + y * w, src_pixels + y * src_stride, w * sizeof(uint32_t));
    }

    out_w = w;
    out_h = h;
    return true;
}

void X11WindowCapture::forward_mouse_click_atomic(double x_1080, double y_1080, int native_w, int native_h, int button, bool is_press) {
    if (!display_ || target_window_ == 0 || current_w_ <= 0 || current_h_ <= 0) return;

    int local_x = std::clamp(static_cast<int>((x_1080 / native_w) * current_w_), 0, current_w_ - 1);
    int local_y = std::clamp(static_cast<int>((y_1080 / native_h) * current_h_), 0, current_h_ - 1);
    int target_screen_x = win_origin_x_ + local_x;
    int target_screen_y = win_origin_y_ + local_y;

    unsigned int x_btn = (button == 0) ? 1 : ((button == 1) ? 3 : 2);
    XTestFakeMotionEvent(display_, -1, target_screen_x, target_screen_y, CurrentTime);
    XTestFakeButtonEvent(display_, x_btn, is_press ? True : False, CurrentTime);
    XFlush(display_);
}

void X11WindowCapture::forward_key(int scancode, bool is_press) {
    if (!display_ || target_window_ == 0 || scancode <= 0) return;
    XTestFakeKeyEvent(display_, scancode, is_press ? True : False, CurrentTime);
    XFlush(display_);
}

std::string X11WindowCapture::get_status_str() const {
    if (target_window_ != 0) return "Attached: " + found_title_ + " (" + std::to_string(current_w_) + "x" + std::to_string(current_h_) + ")";
    return "Searching for TF2 (tf_linux64 / hl2_linux)...";
}
