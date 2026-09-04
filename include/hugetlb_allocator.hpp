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

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <sys/mman.h>

template <typename T, size_t Alignment = 64>
class HugeTlbBuffer {
public:
    HugeTlbBuffer() : ptr_(nullptr), size_(0) {}
    explicit HugeTlbBuffer(size_t count, T val = T()) : ptr_(nullptr), size_(0) {
        allocate(count, val);
    }
    ~HugeTlbBuffer() { free_mem(); }

    HugeTlbBuffer(const HugeTlbBuffer&) = delete;
    HugeTlbBuffer& operator=(const HugeTlbBuffer&) = delete;

    HugeTlbBuffer(HugeTlbBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr;
        o.size_ = 0;
    }
    HugeTlbBuffer& operator=(HugeTlbBuffer&& o) noexcept {
        if (this != &o) {
            free_mem();
            ptr_ = o.ptr_;
            size_ = o.size_;
            o.ptr_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    void allocate(size_t count, T val = T()) {
        free_mem();
        size_ = count;
        size_t bytes = (count * sizeof(T) + Alignment - 1) & ~(Alignment - 1);
        ptr_ = static_cast<T*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0));
        if (ptr_ == MAP_FAILED) {
            if (posix_memalign(reinterpret_cast<void**>(&ptr_), Alignment, bytes) != 0) {
                ptr_ = nullptr;
                return;
            }
        }
        for (size_t i = 0; i < size_; ++i) ptr_[i] = val;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    T& operator[](size_t idx) { return ptr_[idx]; }
    const T& operator[](size_t idx) const { return ptr_[idx]; }

private:
    void free_mem() {
        if (ptr_) {
            size_t bytes = (size_ * sizeof(T) + Alignment - 1) & ~(Alignment - 1);
            if (munmap(ptr_, bytes) != 0) {
                std::free(ptr_);
            }
            ptr_ = nullptr;
            size_ = 0;
        }
    }
    T* ptr_{nullptr};
    size_t size_{0};
};

namespace avx512 {
    template <typename T, size_t Alignment = 64>
    using HugeTlbBuffer = ::HugeTlbBuffer<T, Alignment>;
}
