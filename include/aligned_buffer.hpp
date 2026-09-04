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
#include <cstring>
#include <algorithm>
#include <new>

template <typename T, size_t Alignment = 64>
class AlignedBuffer {
public:
    AlignedBuffer() : m_data(nullptr), m_size(0), m_capacity(0) {}
    
    explicit AlignedBuffer(size_t size, const T& val = T())
        : m_data(nullptr), m_size(0), m_capacity(0)
    {
        resize(size, val);
    }

    ~AlignedBuffer() { free_mem(); }

    AlignedBuffer(const AlignedBuffer& other) : m_data(nullptr), m_size(0), m_capacity(0) {
        if (other.m_size > 0) {
            resize(other.m_size);
            for (size_t i = 0; i < m_size; ++i) m_data[i] = other.m_data[i];
        }
    }

    AlignedBuffer& operator=(const AlignedBuffer& other) {
        if (this != &other) {
            resize(other.m_size);
            for (size_t i = 0; i < m_size; ++i) m_data[i] = other.m_data[i];
        }
        return *this;
    }

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : m_data(other.m_data), m_size(other.m_size), m_capacity(other.m_capacity)
    {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_capacity = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            free_mem();
            m_data = other.m_data;
            m_size = other.m_size;
            m_capacity = other.m_capacity;
            other.m_data = nullptr;
            other.m_size = 0;
            other.m_capacity = 0;
        }
        return *this;
    }

    void resize(size_t new_size, const T& val = T()) {
        if (new_size > m_capacity) {
            size_t alloc_bytes = (new_size * sizeof(T) + Alignment - 1) & ~(Alignment - 1);
            void* ptr = nullptr;
            if (posix_memalign(&ptr, Alignment, alloc_bytes) != 0) {
                throw std::bad_alloc();
            }
            T* new_data = static_cast<T*>(ptr);
            for (size_t i = 0; i < m_size; ++i) new_data[i] = std::move(m_data[i]);
            for (size_t i = m_size; i < new_size; ++i) new_data[i] = val;
            free_mem();
            m_data = new_data;
            m_capacity = new_size;
        } else {
            for (size_t i = m_size; i < new_size; ++i) m_data[i] = val;
        }
        m_size = new_size;
    }

    T* data() noexcept { return m_data; }
    const T* data() const noexcept { return m_data; }
    size_t size() const noexcept { return m_size; }
    T& operator[](size_t idx) { return m_data[idx]; }
    const T& operator[](size_t idx) const { return m_data[idx]; }

private:
    void free_mem() {
        if (m_data) {
            std::free(m_data);
            m_data = nullptr;
        }
    }
    T* m_data{nullptr};
    size_t m_size{0};
    size_t m_capacity{0};
};
