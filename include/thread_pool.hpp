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

#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <emmintrin.h>

class CorePinnedThreadPool {
public:
    using TaskFunc = void(*)(void* user_data, uint32_t thread_id, uint32_t num_threads);

    explicit CorePinnedThreadPool(uint32_t num_threads = 6);
    ~CorePinnedThreadPool();

    CorePinnedThreadPool(const CorePinnedThreadPool&) = delete;
    CorePinnedThreadPool& operator=(const CorePinnedThreadPool&) = delete;

    // 1. Range-based parallel_for: partitions [0, count) across threads and calls func(index, thread_id)
    template <typename F>
    void parallel_for(size_t count, F&& func) {
        if (count == 0) return;
        struct Context {
            F* fn;
            size_t total;
        } ctx { &func, count };

        dispatch_sync([](void* arg, uint32_t tid, uint32_t num_threads) {
            auto* c = static_cast<Context*>(arg);
            const size_t chunk = (c->total + num_threads - 1) / num_threads;
            const size_t start = std::min(static_cast<size_t>(tid) * chunk, c->total);
            const size_t end   = std::min(start + chunk, c->total);
            for (size_t i = start; i < end; ++i) {
                (*c->fn)(i, tid);
            }
        }, &ctx);
    }

    // 2. Raw function pointer dispatch with user data
    void parallel_for(TaskFunc func, void* user_data) {
        dispatch_sync(func, user_data);
    }

    uint32_t thread_count() const noexcept { return m_num_threads; }

private:
    void dispatch_sync(TaskFunc func, void* user_data);
    void worker_loop(uint32_t thread_id);
    void pin_thread(uint32_t core_id);

    uint32_t m_num_threads;
    std::vector<std::thread> m_workers;

    alignas(64) std::atomic<uint32_t> m_task_generation{0};
    alignas(64) std::atomic<uint32_t> m_completed_workers{0};
    alignas(64) std::atomic<bool>     m_shutdown{false};

    TaskFunc m_current_task{nullptr};
    void*    m_current_user_data{nullptr};
};

// Aliases for Coprocessor Engine namespace
namespace avx512 {
    using CorePinnedThreadPool = ::CorePinnedThreadPool;
    using LockFreeThreadPool   = ::CorePinnedThreadPool;
}
