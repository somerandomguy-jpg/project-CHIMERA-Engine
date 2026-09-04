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

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <array>
#include <cstdint>
#include <string>

class VulkanRenderer {
public:
    VulkanRenderer(GLFWwindow* window, int width, int height);
    ~VulkanRenderer();

    void toggle_vsync();
    void wait_slot_ready(uint32_t slot, uint64_t timeout_ns);
    void* get_mapped_ptr(uint32_t slot) noexcept { return m_staging_mapped[slot % 3]; }
    void* get_mapped_buffer(uint32_t slot) noexcept { return get_mapped_ptr(slot); }
    void render_frame(uint32_t slot, uint64_t frame_index);

private:
    void init_vulkan(GLFWwindow* window);
    void cleanup_vulkan();
    void create_swapchain();
    void cleanup_swapchain();
    void create_image_resources();
    void create_pipeline();
    void create_command_buffers();
    void create_sync_objects();

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);

    int m_width;
    int m_height;
    bool m_vsync{false};
    GLFWwindow* m_window{nullptr};

    VkInstance m_instance{VK_NULL_HANDLE};
    VkSurfaceKHR m_surface{VK_NULL_HANDLE};
    VkPhysicalDevice m_physical_device{VK_NULL_HANDLE};
    VkDevice m_device{VK_NULL_HANDLE};
    VkQueue m_graphics_queue{VK_NULL_HANDLE};
    VkQueue m_present_queue{VK_NULL_HANDLE};
    uint32_t m_queue_family_index{0};

    VkSwapchainKHR m_swapchain{VK_NULL_HANDLE};
    VkFormat m_swapchain_format{VK_FORMAT_B8G8R8A8_UNORM};
    VkExtent2D m_swapchain_extent{};
    std::vector<VkImage> m_swapchain_images;
    std::vector<VkImageView> m_swapchain_image_views;
    std::vector<VkFramebuffer> m_framebuffers;

    VkRenderPass m_render_pass{VK_NULL_HANDLE};
    VkDescriptorSetLayout m_descriptor_layout{VK_NULL_HANDLE};
    VkDescriptorPool m_descriptor_pool{VK_NULL_HANDLE};
    VkDescriptorSet m_descriptor_set{VK_NULL_HANDLE};
    VkPipelineLayout m_pipeline_layout{VK_NULL_HANDLE};
    VkPipeline m_pipeline{VK_NULL_HANDLE};

    VkCommandPool m_command_pool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> m_command_buffers;

    // Triple-buffered Staging buffers for CPU -> GPU zero-copy DMA
    std::array<VkBuffer, 3> m_staging_buffers{};
    std::array<VkDeviceMemory, 3> m_staging_memory{};
    std::array<void*, 3> m_staging_mapped{nullptr, nullptr, nullptr};

    // Render target texture on GPU
    VkImage m_render_image{VK_NULL_HANDLE};
    VkDeviceMemory m_render_image_memory{VK_NULL_HANDLE};
    VkImageView m_render_image_view{VK_NULL_HANDLE};
    VkSampler m_sampler{VK_NULL_HANDLE};

    std::array<VkSemaphore, 3> m_image_available_semaphores{};
    std::array<VkSemaphore, 3> m_render_finished_semaphores{};
    std::array<VkFence, 3> m_in_flight_fences{};
};
