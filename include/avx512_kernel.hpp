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
#ifndef AVX512_KERNEL_HPP
#define AVX512_KERNEL_HPP

#include <cstdint>
#include <vector>
#include "thread_pool.hpp"
#include "aligned_buffer.hpp"
#include "hugetlb_allocator.hpp"

enum class UpscaleMode {
    TieredHyperOmniV4,         // [FLAGSHIP V4] 4x Quad-Stream Unrolled + Zero-Copy Temporal
    TieredHyperOmniV3,         // Tiered Omni V3 (2x Unrolled)
    BioPhysarumSlimeMold2x,    // Physarum Slime Mold Flux
    BioQuantumIsingGlass2x,    // Quantum Ising Spin Ground State
    BioNeuromorphicLif2x,      // Neuromorphic LIF Spiking Vector Mesh
    BioGfniGaloisRecon2x       // GFNI Galois Field Recon
};

class Avx512Upscaler {
public:
    // --- FLAGSHIP 4X QUAD-STREAM PIPELINE ---
    static void upscale_tiered_hyper_omni_v4(
        const uint32_t* src, uint32_t* dst,
        int src_w, int src_h, int dst_w, int dst_h,
        uint64_t frame_index, CorePinnedThreadPool& pool);

    static void upscale_tiered_hyper_omni_v3(const uint32_t* src, const uint32_t* prev_frame, uint32_t* dst, int src_w, int src_h, int dst_w, int dst_h, uint64_t frame_index, CorePinnedThreadPool& pool);
    static void upscale_physarum_slime_mold_2x(const uint32_t* src, uint32_t* dst, int src_w, int src_h, int dst_w, int dst_h, CorePinnedThreadPool& pool);
    static void upscale_quantum_ising_glass_2x(const uint32_t* src, uint32_t* dst, int src_w, int src_h, int dst_w, int dst_h, CorePinnedThreadPool& pool);
    static void upscale_neuromorphic_lif_2x(const uint32_t* src, uint32_t* dst, int src_w, int src_h, int dst_w, int dst_h, CorePinnedThreadPool& pool);
    static void upscale_gfni_galois_recon_2x(const uint32_t* src, uint32_t* dst, int src_w, int src_h, int dst_w, int dst_h, CorePinnedThreadPool& pool);
};
#endif
