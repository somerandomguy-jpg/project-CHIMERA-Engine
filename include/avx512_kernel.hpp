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
