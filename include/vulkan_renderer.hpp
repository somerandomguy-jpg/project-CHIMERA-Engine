#pragma once
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <cstdint>

struct VulkanTimings {
    double acquire_us{0.0};
    double record_us{0.0};
    double submit_us{0.0};
    double present_us{0.0};
};

class VulkanRenderer {
public:
    VulkanRenderer(GLFWwindow* window, int width, int height);
    ~VulkanRenderer();

    uint32_t* get_staging_ptr(size_t slot) { return mapped_staging_[slot]; }
    double wait_slot_ready(size_t slot, uint64_t frame_idx);
    VulkanTimings render_frame(size_t slot, uint64_t frame_idx);

    void toggle_vsync();
    void set_present_mode(VkPresentModeKHR mode);
    bool is_vsync_enabled() const { return current_present_mode_ == VK_PRESENT_MODE_FIFO_KHR; }
    const char* get_present_mode_str() const;

    int width() const { return width_; }
    int height() const { return height_; }

private:
    void init_vulkan();
    void create_swapchain();
    void recreate_swapchain();
    void cleanup_swapchain();
    void create_images();
    void create_pipeline();
    void create_pinned_buffers();
    void create_sync_objects();
    uint32_t find_memory_type(uint32_t filter, VkMemoryPropertyFlags props);

    GLFWwindow* window_;
    int width_, height_;

    VkInstance instance_{VK_NULL_HANDLE};
    VkSurfaceKHR surface_{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueue graphics_queue_{VK_NULL_HANDLE};
    uint32_t graphics_family_{0};

    VkSwapchainKHR swapchain_{VK_NULL_HANDLE};
    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_views_;
    VkFormat swapchain_format_{VK_FORMAT_B8G8R8A8_UNORM};
    VkExtent2D swapchain_extent_{};
    VkPresentModeKHR current_present_mode_{VK_PRESENT_MODE_FIFO_KHR};

    static constexpr size_t RING_SIZE = 3;

    VkImage gpu_images_[RING_SIZE]{};
    VkDeviceMemory gpu_image_memories_[RING_SIZE]{};
    VkImageView gpu_image_views_[RING_SIZE]{};
    VkSampler sampler_{VK_NULL_HANDLE};

    VkRenderPass render_pass_{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline pipeline_{VK_NULL_HANDLE};
    VkDescriptorSetLayout desc_layout_{VK_NULL_HANDLE};
    VkDescriptorPool desc_pool_{VK_NULL_HANDLE};
    VkDescriptorSet desc_sets_[RING_SIZE]{};
    std::vector<VkFramebuffer> framebuffers_;

    VkBuffer staging_buffers_[RING_SIZE]{};
    VkDeviceMemory staging_memory_[RING_SIZE]{};
    uint32_t* mapped_staging_[RING_SIZE]{};

    VkCommandPool graphics_pool_{VK_NULL_HANDLE};
    VkCommandBuffer graphics_cmds_[RING_SIZE]{};

    VkSemaphore timeline_semaphore_{VK_NULL_HANDLE};
    VkSemaphore image_available_sem_[RING_SIZE]{};
    VkSemaphore render_finished_sem_[RING_SIZE]{};
};
