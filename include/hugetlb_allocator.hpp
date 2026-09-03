#pragma once
#ifndef HUGETLB_ALLOCATOR_HPP
#define HUGETLB_ALLOCATOR_HPP

#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <new>
#include <sys/mman.h>
#include <unistd.h>

template <typename T, size_t Alignment = 64>
class HugeTlbBuffer {
public:
    explicit HugeTlbBuffer(size_t count = 0, T val = T{}) : count_(count) {
        if (count_ > 0) allocate(count_, val);
    }

    ~HugeTlbBuffer() {
        if (ptr_) {
            if (is_hugetlb_) {
                munmap(ptr_, allocated_bytes_);
            } else {
                free(ptr_);
            }
        }
    }

    void resize(size_t count, T val = T{}) {
        if (count == count_) return;
        if (ptr_) {
            if (is_hugetlb_) munmap(ptr_, allocated_bytes_);
            else free(ptr_);
        }
        count_ = count;
        if (count_ > 0) allocate(count_, val);
        else { ptr_ = nullptr; allocated_bytes_ = 0; is_hugetlb_ = false; }
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return count_; }
    T& operator[](size_t idx) noexcept { return ptr_[idx]; }
    const T& operator[](size_t idx) const noexcept { return ptr_[idx]; }

    HugeTlbBuffer(const HugeTlbBuffer&) = delete;
    HugeTlbBuffer& operator=(const HugeTlbBuffer&) = delete;

private:
    void allocate(size_t count, T val) {
        size_t bytes = count * sizeof(T);
        // Align to 2MB boundary for HugeTLB
        size_t size_2mb = (bytes + (2 * 1024 * 1024 - 1)) & ~(2 * 1024 * 1024 - 1);
        
        #ifdef MAP_HUGETLB
        void* p = mmap(nullptr, size_2mb, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED) {
            ptr_ = static_cast<T*>(p);
            allocated_bytes_ = size_2mb;
            is_hugetlb_ = true;
            std::fill_n(ptr_, count_, val);
            return;
        }
        #endif

        // Fallback: 64-byte posix_memalign
        size_t aligned_bytes = (bytes + Alignment - 1) & ~(Alignment - 1);
        if (posix_memalign(reinterpret_cast<void**>(&ptr_), Alignment, aligned_bytes) != 0) {
            ptr_ = nullptr;
            throw std::bad_alloc();
        }
        allocated_bytes_ = aligned_bytes;
        is_hugetlb_ = false;
        std::fill_n(ptr_, count_, val);
    }

    T* ptr_{nullptr};
    size_t count_{0};
    size_t allocated_bytes_{0};
    bool is_hugetlb_{false};
};

#endif
