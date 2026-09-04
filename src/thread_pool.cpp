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

#include "thread_pool.hpp"
#include <pthread.h>

CorePinnedThreadPool::CorePinnedThreadPool(uint32_t num_threads)
    : m_num_threads(num_threads)
{
    // Pin calling master thread to Core 0
    pin_thread(0);

    // Spawn workers for Cores 1 .. (num_threads - 1)
    m_workers.reserve(num_threads - 1);
    for (uint32_t i = 1; i < num_threads; ++i) {
        m_workers.emplace_back(&CorePinnedThreadPool::worker_loop, this, i);
    }
}

CorePinnedThreadPool::~CorePinnedThreadPool() {
    m_shutdown.store(true, std::memory_order_release);
    m_task_generation.fetch_add(1, std::memory_order_release);

    for (auto& worker : m_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void CorePinnedThreadPool::pin_thread(uint32_t core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void CorePinnedThreadPool::worker_loop(uint32_t thread_id) {
    pin_thread(thread_id);
    uint32_t local_generation = 0;

    while (true) {
        // Spin-wait on task generation token
        while (m_task_generation.load(std::memory_order_acquire) == local_generation) {
            if (m_shutdown.load(std::memory_order_relaxed)) return;
            _mm_pause();
        }

        local_generation = m_task_generation.load(std::memory_order_acquire);
        if (m_shutdown.load(std::memory_order_relaxed)) return;

        if (m_current_task) {
            m_current_task(m_current_user_data, thread_id, m_num_threads);
        }

        m_completed_workers.fetch_add(1, std::memory_order_release);
    }
}

void CorePinnedThreadPool::dispatch_sync(TaskFunc func, void* user_data) {
    m_current_task = func;
    m_current_user_data = user_data;
    m_completed_workers.store(0, std::memory_order_relaxed);

    // Release task to Workers 1..5
    m_task_generation.fetch_add(1, std::memory_order_release);

    // Master executes Worker 0 slice on Core 0
    func(user_data, 0, m_num_threads);

    // Spin-wait until Workers 1..5 finish
    const uint32_t expected = m_num_threads - 1;
    while (m_completed_workers.load(std::memory_order_acquire) < expected) {
        _mm_pause();
    }
}
