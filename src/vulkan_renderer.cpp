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

#include "vulkan_renderer.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace {
    std::vector<char> read_shader_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            file.open("shaders/" + filename, std::ios::ate | std::ios::binary);
        }
        if (!file.is_open()) {
            throw std::runtime_error("Failed to find shader binary: " + filename);
        }
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        return buffer;
    }
}

VulkanRenderer::VulkanRenderer(GLFWwindow* window, int width, int height)
    : m_width(width), m_height(height), m_window(window)
{
    init_vulkan(window);
}

VulkanRenderer::~VulkanRenderer() {
    cleanup_vulkan();
}

void VulkanRenderer::toggle_vsync() {
    m_vsync = !m_vsync;
    vkDeviceWaitIdle(m_device);
    cleanup_swapchain();
    create_swapchain();
}

uint32_t VulkanRenderer::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(m_physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

void VulkanRenderer::init_vulkan(GLFWwindow* window) {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "AVX-512 Total CPU Engine";
    app_info.apiVersion = VK_API_VERSION_1_2;

    uint32_t glfw_ext_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    VkInstanceCreateInfo inst_info{};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledExtensionCount = glfw_ext_count;
    inst_info.ppEnabledExtensionNames = glfw_exts;

    if (vkCreateInstance(&inst_info, nullptr, &m_instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }

    if (glfwCreateWindowSurface(m_instance, window, nullptr, &m_surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface!");
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(m_instance, &device_count, nullptr);
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(m_instance, &device_count, devices.data());

    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(device, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            m_physical_device = device;
            std::cout << "[INFO] Selected Discrete GPU: " << props.deviceName << "
";
            break;
        }
    }
    if (m_physical_device == VK_NULL_HANDLE) m_physical_device = devices[0];

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = 0;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    const char* device_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo dev_info{};
    dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_info.queueCreateInfoCount = 1;
    dev_info.pQueueCreateInfos = &queue_info;
    dev_info.enabledExtensionCount = 1;
    dev_info.ppEnabledExtensionNames = device_exts;

    if (vkCreateDevice(m_physical_device, &dev_info, nullptr, &m_device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device!");
    }

    vkGetDeviceQueue(m_device, 0, 0, &m_graphics_queue);
    m_present_queue = m_graphics_queue;

    create_swapchain();
    create_image_resources();
    create_pipeline();
    create_command_buffers();
    create_sync_objects();
}

void VulkanRenderer::create_swapchain() {
    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = m_surface;
    create_info.minImageCount = 3;
    create_info.imageFormat = m_swapchain_format;
    create_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    create_info.imageExtent = VkExtent2D{static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height)};
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = m_vsync ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_IMMEDIATE_KHR;
    create_info.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(m_device, &create_info, nullptr, &m_swapchain) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swapchain!");
    }

    uint32_t image_count = 0;
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &image_count, nullptr);
    m_swapchain_images.resize(image_count);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &image_count, m_swapchain_images.data());

    m_swapchain_image_views.resize(image_count);
    for (size_t i = 0; i < image_count; ++i) {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = m_swapchain_images[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = m_swapchain_format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;
        vkCreateImageView(m_device, &view_info, nullptr, &m_swapchain_image_views[i]);
    }
}

void VulkanRenderer::cleanup_swapchain() {
    for (auto fb : m_framebuffers) vkDestroyFramebuffer(m_device, fb, nullptr);
    for (auto iv : m_swapchain_image_views) vkDestroyImageView(m_device, iv, nullptr);
    if (m_swapchain) vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
}

void VulkanRenderer::create_image_resources() {
    VkDeviceSize buffer_size = m_width * m_height * sizeof(uint32_t);

    for (int i = 0; i < 3; ++i) {
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = buffer_size;
        buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCreateBuffer(m_device, &buf_info, nullptr, &m_staging_buffers[i]);

        VkMemoryRequirements mem_reqs{};
        vkGetBufferMemoryRequirements(m_device, m_staging_buffers[i], &mem_reqs);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_reqs.size;
        alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkAllocateMemory(m_device, &alloc_info, nullptr, &m_staging_memory[i]);
        vkBindBufferMemory(m_device, m_staging_buffers[i], m_staging_memory[i], 0);
        vkMapMemory(m_device, m_staging_memory[i], 0, buffer_size, 0, &m_staging_mapped[i]);
    }

    VkImageCreateInfo img_info{};
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType = VK_IMAGE_TYPE_2D;
    img_info.extent = VkExtent3D{static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height), 1};
    img_info.mipLevels = 1;
    img_info.arrayLayers = 1;
    img_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    img_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    img_info.samples = VK_SAMPLE_COUNT_1_BIT;

    vkCreateImage(m_device, &img_info, nullptr, &m_render_image);

    VkMemoryRequirements mem_reqs{};
    vkGetImageMemoryRequirements(m_device, m_render_image, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vkAllocateMemory(m_device, &alloc_info, nullptr, &m_render_image_memory);
    vkBindImageMemory(m_device, m_render_image, m_render_image_memory, 0);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = m_render_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    vkCreateImageView(m_device, &view_info, nullptr, &m_render_image_view);

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    vkCreateSampler(m_device, &sampler_info, nullptr, &m_sampler);
}

void VulkanRenderer::create_pipeline() {
    auto vert_code = read_shader_file("fullscreen.vert.spv");
    auto frag_code = read_shader_file("fullscreen.frag.spv");

    VkShaderModuleCreateInfo vert_info{};
    vert_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vert_info.codeSize = vert_code.size();
    vert_info.pCode = reinterpret_cast<const uint32_t*>(vert_code.data());
    VkShaderModule vert_module;
    vkCreateShaderModule(m_device, &vert_info, nullptr, &vert_module);

    VkShaderModuleCreateInfo frag_info{};
    frag_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    frag_info.codeSize = frag_code.size();
    frag_info.pCode = reinterpret_cast<const uint32_t*>(frag_code.data());
    VkShaderModule frag_module;
    vkCreateShaderModule(m_device, &frag_info, nullptr, &frag_module);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_module;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_module;
    stages[1].pName = "main";

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;
    vkCreateDescriptorSetLayout(m_device, &layout_info, nullptr, &m_descriptor_layout);

    VkPipelineLayoutCreateInfo pipe_layout_info{};
    pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe_layout_info.setLayoutCount = 1;
    pipe_layout_info.pSetLayouts = &m_descriptor_layout;
    vkCreatePipelineLayout(m_device, &pipe_layout_info, nullptr, &m_pipeline_layout);

    VkAttachmentDescription color_att{};
    color_att.format = m_swapchain_format;
    color_att.samples = VK_SAMPLE_COUNT_1_BIT;
    color_att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference att_ref{};
    att_ref.attachment = 0;
    att_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &att_ref;

    VkRenderPassCreateInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 1;
    rp_info.pAttachments = &color_att;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    vkCreateRenderPass(m_device, &rp_info, nullptr, &m_render_pass);

    VkPipelineVertexInputStateCreateInfo v_input{};
    v_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo i_asm{};
    i_asm.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    i_asm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{0.0f, 0.0f, static_cast<float>(m_width), static_cast<float>(m_height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height)}};
    VkPipelineViewportStateCreateInfo vp_state{};
    vp_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp_state.viewportCount = 1;
    vp_state.pViewports = &viewport;
    vp_state.scissorCount = 1;
    vp_state.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blend_att{};
    blend_att.colorWriteMask = 0xF;
    blend_att.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_att;

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &v_input;
    pipeline_info.pInputAssemblyState = &i_asm;
    pipeline_info.pViewportState = &vp_state;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState = &ms;
    pipeline_info.pColorBlendState = &blend;
    pipeline_info.layout = m_pipeline_layout;
    pipeline_info.renderPass = m_render_pass;

    vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &m_pipeline);

    vkDestroyShaderModule(m_device, vert_module, nullptr);
    vkDestroyShaderModule(m_device, frag_module, nullptr);

    m_framebuffers.resize(m_swapchain_image_views.size());
    for (size_t i = 0; i < m_swapchain_image_views.size(); ++i) {
        VkFramebufferCreateInfo fb_info{};
        fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass = m_render_pass;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments = &m_swapchain_image_views[i];
        fb_info.width = m_width;
        fb_info.height = m_height;
        fb_info.layers = 1;
        vkCreateFramebuffer(m_device, &fb_info, nullptr, &m_framebuffers[i]);
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_descriptor_pool);

    VkDescriptorSetAllocateInfo ds_alloc{};
    ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_alloc.descriptorPool = m_descriptor_pool;
    ds_alloc.descriptorSetCount = 1;
    ds_alloc.pSetLayouts = &m_descriptor_layout;
    vkAllocateDescriptorSets(m_device, &ds_alloc, &m_descriptor_set);

    VkDescriptorImageInfo img_desc{};
    img_desc.sampler = m_sampler;
    img_desc.imageView = m_render_image_view;
    img_desc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write_ds{};
    write_ds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_ds.dstSet = m_descriptor_set;
    write_ds.dstBinding = 0;
    write_ds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write_ds.descriptorCount = 1;
    write_ds.pImageInfo = &img_desc;
    vkUpdateDescriptorSets(m_device, 1, &write_ds, 0, nullptr);
}

void VulkanRenderer::create_command_buffers() {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(m_device, &pool_info, nullptr, &m_command_pool);

    m_command_buffers.resize(3);
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = m_command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 3;
    vkAllocateCommandBuffers(m_device, &alloc_info, m_command_buffers.data());
}

void VulkanRenderer::create_sync_objects() {
    for (int i = 0; i < 3; ++i) {
        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        vkCreateSemaphore(m_device, &sem_info, nullptr, &m_image_available_semaphores[i]);
        vkCreateSemaphore(m_device, &sem_info, nullptr, &m_render_finished_semaphores[i]);
        vkCreateFence(m_device, &fence_info, nullptr, &m_in_flight_fences[i]);
    }
}

void VulkanRenderer::wait_slot_ready(uint32_t slot, uint64_t timeout_ns) {
    vkWaitForFences(m_device, 1, &m_in_flight_fences[slot % 3], VK_TRUE, timeout_ns);
    vkResetFences(m_device, 1, &m_in_flight_fences[slot % 3]);
}

void VulkanRenderer::render_frame(uint32_t slot, uint64_t /*frame_index*/) {
    uint32_t s = slot % 3;
    uint32_t image_index = 0;
    vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX, m_image_available_semaphores[s], VK_NULL_HANDLE, &image_index);

    VkCommandBuffer cmd = m_command_buffers[s];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin_info);

    VkImageMemoryBarrier barrier_to_dst{};
    barrier_to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier_to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier_to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_to_dst.image = m_render_image;
    barrier_to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier_to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier_to_dst);

    VkBufferImageCopy copy_region{};
    copy_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.imageExtent = VkExtent3D{static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height), 1};

    vkCmdCopyBufferToImage(cmd, m_staging_buffers[s], m_render_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    VkImageMemoryBarrier barrier_to_read{};
    barrier_to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier_to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_to_read.image = m_render_image;
    barrier_to_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier_to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier_to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier_to_read);

    VkRenderPassBeginInfo rp_begin{};
    rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin.renderPass = m_render_pass;
    rp_begin.framebuffer = m_framebuffers[image_index];
    rp_begin.renderArea = {{0, 0}, {static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height)}};

    vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout, 0, 1, &m_descriptor_set, 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);

    vkEndCommandBuffer(cmd);

    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &m_image_available_semaphores[s];
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &m_render_finished_semaphores[s];

    vkQueueSubmit(m_graphics_queue, 1, &submit_info, m_in_flight_fences[s]);

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &m_render_finished_semaphores[s];
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &m_swapchain;
    present_info.pImageIndices = &image_index;

    vkQueuePresentKHR(m_present_queue, &present_info);
}

void VulkanRenderer::cleanup_vulkan() {
    vkDeviceWaitIdle(m_device);

    for (int i = 0; i < 3; ++i) {
        if (m_staging_mapped[i]) vkUnmapMemory(m_device, m_staging_memory[i]);
        if (m_staging_buffers[i]) vkDestroyBuffer(m_device, m_staging_buffers[i], nullptr);
        if (m_staging_memory[i]) vkFreeMemory(m_device, m_staging_memory[i], nullptr);

        if (m_image_available_semaphores[i]) vkDestroySemaphore(m_device, m_image_available_semaphores[i], nullptr);
        if (m_render_finished_semaphores[i]) vkDestroySemaphore(m_device, m_render_finished_semaphores[i], nullptr);
        if (m_in_flight_fences[i]) vkDestroyFence(m_device, m_in_flight_fences[i], nullptr);
    }

    if (m_sampler) vkDestroySampler(m_device, m_sampler, nullptr);
    if (m_render_image_view) vkDestroyImageView(m_device, m_render_image_view, nullptr);
    if (m_render_image) vkDestroyImage(m_device, m_render_image, nullptr);
    if (m_render_image_memory) vkFreeMemory(m_device, m_render_image_memory, nullptr);

    if (m_command_pool) vkDestroyCommandPool(m_device, m_command_pool, nullptr);
    if (m_pipeline) vkDestroyPipeline(m_device, m_pipeline, nullptr);
    if (m_pipeline_layout) vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
    if (m_descriptor_pool) vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
    if (m_descriptor_layout) vkDestroyDescriptorSetLayout(m_device, m_descriptor_layout, nullptr);
    if (m_render_pass) vkDestroyRenderPass(m_device, m_render_pass, nullptr);

    cleanup_swapchain();

    if (m_device) vkDestroyDevice(m_device, nullptr);
    if (m_surface) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
}
