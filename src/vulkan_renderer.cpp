#include "vulkan_renderer.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <chrono>

static std::vector<char> read_file(const std::string& filename) {
    std::vector<std::string> paths = { filename, "../" + filename, "shaders/" + filename, "../shaders/" + filename };
    for (const auto& path : paths) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (file.is_open()) {
            size_t size = static_cast<size_t>(file.tellg());
            std::vector<char> buf(size);
            file.seekg(0);
            file.read(buf.data(), size);
            return buf;
        }
    }
    throw std::runtime_error("Failed to find shader binary: " + filename);
}

VulkanRenderer::VulkanRenderer(GLFWwindow* window, int width, int height)
: window_(window), width_(width), height_(height) {
    init_vulkan();
    create_swapchain();
    create_images();
    create_pipeline();
    create_pinned_buffers();
    create_sync_objects();
}

VulkanRenderer::~VulkanRenderer() {
    vkDeviceWaitIdle(device_);
    for (size_t i = 0; i < RING_SIZE; ++i) {
        vkDestroyBuffer(device_, staging_buffers_[i], nullptr);
        vkFreeMemory(device_, staging_memory_[i], nullptr);
        vkDestroyImageView(device_, gpu_image_views_[i], nullptr);
        vkDestroyImage(device_, gpu_images_[i], nullptr);
        vkFreeMemory(device_, gpu_image_memories_[i], nullptr);
        vkDestroySemaphore(device_, image_available_sem_[i], nullptr);
        vkDestroySemaphore(device_, render_finished_sem_[i], nullptr);
    }
    vkDestroySampler(device_, sampler_, nullptr);
    vkDestroySemaphore(device_, timeline_semaphore_, nullptr);
    vkDestroyCommandPool(device_, graphics_pool_, nullptr);
    for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
    vkDestroyPipeline(device_, pipeline_, nullptr);
    vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    vkDestroyRenderPass(device_, render_pass_, nullptr);
    vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
    vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr);
    for (auto iv : swapchain_views_) vkDestroyImageView(device_, iv, nullptr);
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    vkDestroyDevice(device_, nullptr);
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkDestroyInstance(instance_, nullptr);
}

uint32_t VulkanRenderer::find_memory_type(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

void VulkanRenderer::init_vulkan() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "AVX-512 Bio-Physics Super-Resolution Engine";
    app_info.apiVersion = VK_API_VERSION_1_2;

    uint32_t glfw_ext_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    VkInstanceCreateInfo inst_info{};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledExtensionCount = glfw_ext_count;
    inst_info.ppEnabledExtensionNames = glfw_exts;

    if (vkCreateInstance(&inst_info, nullptr, &instance_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }

    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface!");
    }

    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance_, &dev_count, nullptr);
    if (dev_count == 0) throw std::runtime_error("No Vulkan physical devices found!");
    std::vector<VkPhysicalDevice> devs(dev_count);
    vkEnumeratePhysicalDevices(instance_, &dev_count, devs.data());

    physical_device_ = devs[0];
    for (const auto& dev : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = dev;
            std::cout << "[INFO] Selected Discrete GPU: " << props.deviceName << std::endl;
            break;
        }
    }

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, qf_props.data());

    graphics_family_ = 0;
    for (uint32_t i = 0; i < qf_count; ++i) {
        if (qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphics_family_ = i;
            break;
        }
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo gq_info{};
    gq_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    gq_info.queueFamilyIndex = graphics_family_;
    gq_info.queueCount = 1;
    gq_info.pQueuePriorities = &priority;

    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feat{};
    timeline_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    timeline_feat.timelineSemaphore = VK_TRUE;

    const char* dev_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo dev_info{};
    dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_info.pNext = &timeline_feat;
    dev_info.queueCreateInfoCount = 1;
    dev_info.pQueueCreateInfos = &gq_info;
    dev_info.enabledExtensionCount = 1;
    dev_info.ppEnabledExtensionNames = dev_exts;

    if (vkCreateDevice(physical_device_, &dev_info, nullptr, &device_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device!");
    }
    vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
}

void VulkanRenderer::create_swapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &caps);

    swapchain_format_ = VK_FORMAT_B8G8R8A8_UNORM;
    swapchain_extent_ = {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)};

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR sc_info{};
    sc_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sc_info.surface = surface_;
    sc_info.minImageCount = image_count;
    sc_info.imageFormat = swapchain_format_;
    sc_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sc_info.imageExtent = swapchain_extent_;
    sc_info.imageArrayLayers = 1;
    sc_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sc_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sc_info.preTransform = caps.currentTransform;
    sc_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sc_info.presentMode = current_present_mode_;
    sc_info.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device_, &sc_info, nullptr, &swapchain_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swapchain!");
    }

    uint32_t img_count = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &img_count, nullptr);
    swapchain_images_.resize(img_count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &img_count, swapchain_images_.data());

    swapchain_views_.resize(img_count);
    for (size_t i = 0; i < img_count; ++i) {
        VkImageViewCreateInfo iv_info{};
        iv_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv_info.image = swapchain_images_[i];
        iv_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv_info.format = swapchain_format_;
        iv_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(device_, &iv_info, nullptr, &swapchain_views_[i]);
    }
}

void VulkanRenderer::cleanup_swapchain() {
    for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
    framebuffers_.clear();
    for (auto iv : swapchain_views_) vkDestroyImageView(device_, iv, nullptr);
    swapchain_views_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VulkanRenderer::recreate_swapchain() {
    vkDeviceWaitIdle(device_);
    cleanup_swapchain();
    create_swapchain();

    framebuffers_.resize(swapchain_views_.size());
    for (size_t i = 0; i < swapchain_views_.size(); ++i) {
        VkFramebufferCreateInfo fb_info{};
        fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass = render_pass_;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments = &swapchain_views_[i];
        fb_info.width = width_;
        fb_info.height = height_;
        fb_info.layers = 1;
        vkCreateFramebuffer(device_, &fb_info, nullptr, &framebuffers_[i]);
    }
}

void VulkanRenderer::set_present_mode(VkPresentModeKHR mode) {
    if (current_present_mode_ == mode) return;
    current_present_mode_ = mode;
    recreate_swapchain();
    std::cout << "[INFO] Present mode switched to: " << get_present_mode_str() << std::endl;
}

void VulkanRenderer::toggle_vsync() {
    if (current_present_mode_ == VK_PRESENT_MODE_FIFO_KHR) {
        set_present_mode(VK_PRESENT_MODE_MAILBOX_KHR);
    } else {
        set_present_mode(VK_PRESENT_MODE_FIFO_KHR);
    }
}

const char* VulkanRenderer::get_present_mode_str() const {
    switch (current_present_mode_) {
        case VK_PRESENT_MODE_FIFO_KHR: return "FIFO (VSync 60 FPS)";
        case VK_PRESENT_MODE_MAILBOX_KHR: return "MAILBOX (Uncapped Low-Latency)";
        case VK_PRESENT_MODE_IMMEDIATE_KHR: return "IMMEDIATE (Uncapped)";
        default: return "UNKNOWN";
    }
}

void VulkanRenderer::create_images() {
    VkSamplerCreateInfo samp_info{};
    samp_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samp_info.magFilter = VK_FILTER_LINEAR;
    samp_info.minFilter = VK_FILTER_LINEAR;
    samp_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(device_, &samp_info, nullptr, &sampler_);

    for (size_t i = 0; i < RING_SIZE; ++i) {
        VkImageCreateInfo img_info{};
        img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.format = VK_FORMAT_B8G8R8A8_UNORM;
        img_info.extent = {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_), 1};
        img_info.mipLevels = 1;
        img_info.arrayLayers = 1;
        img_info.samples = VK_SAMPLE_COUNT_1_BIT;
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        vkCreateImage(device_, &img_info, nullptr, &gpu_images_[i]);

        VkMemoryRequirements mem_reqs;
        vkGetImageMemoryRequirements(device_, gpu_images_[i], &mem_reqs);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_reqs.size;
        alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device_, &alloc_info, nullptr, &gpu_image_memories_[i]);
        vkBindImageMemory(device_, gpu_images_[i], gpu_image_memories_[i], 0);

        VkImageViewCreateInfo iv_info{};
        iv_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv_info.image = gpu_images_[i];
        iv_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv_info.format = VK_FORMAT_B8G8R8A8_UNORM;
        iv_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(device_, &iv_info, nullptr, &gpu_image_views_[i]);
    }
}

void VulkanRenderer::create_pinned_buffers() {
    VkDeviceSize size = width_ * height_ * 4;

    VkCommandPoolCreateInfo cp_info{};
    cp_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp_info.queueFamilyIndex = graphics_family_;
    vkCreateCommandPool(device_, &cp_info, nullptr, &graphics_pool_);

    for (size_t i = 0; i < RING_SIZE; ++i) {
        VkBufferCreateInfo b_info{};
        b_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        b_info.size = size;
        b_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        b_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device_, &b_info, nullptr, &staging_buffers_[i]);

        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(device_, staging_buffers_[i], &reqs);

        VkMemoryAllocateInfo a_info{};
        a_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        a_info.allocationSize = reqs.size;
        a_info.memoryTypeIndex = find_memory_type(reqs.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device_, &a_info, nullptr, &staging_memory_[i]);
        vkBindBufferMemory(device_, staging_buffers_[i], staging_memory_[i], 0);

        vkMapMemory(device_, staging_memory_[i], 0, size, 0, reinterpret_cast<void**>(&mapped_staging_[i]));

        VkCommandBufferAllocateInfo cb_info{};
        cb_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cb_info.commandPool = graphics_pool_;
        cb_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cb_info.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &cb_info, &graphics_cmds_[i]);
    }
}

void VulkanRenderer::create_pipeline() {
    auto vert_code = read_file("fullscreen.vert.spv");
    auto frag_code = read_file("fullscreen.frag.spv");

    auto create_shader_module = [this](const std::vector<char>& code) {
        VkShaderModuleCreateInfo sm_info{};
        sm_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_info.codeSize = code.size();
        sm_info.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule mod;
        vkCreateShaderModule(device_, &sm_info, nullptr, &mod);
        return mod;
    };

    VkShaderModule vert_mod = create_shader_module(vert_code);
    VkShaderModule frag_mod = create_shader_module(frag_code);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_mod;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_mod;
    stages[1].pName = "main";

    VkAttachmentDescription att{};
    att.format = swapchain_format_;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference att_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &att_ref;

    VkRenderPassCreateInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 1;
    rp_info.pAttachments = &att;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    vkCreateRenderPass(device_, &rp_info, nullptr, &render_pass_);

    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_info{};
    dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_info.bindingCount = 1;
    dsl_info.pBindings = &b;
    vkCreateDescriptorSetLayout(device_, &dsl_info, nullptr, &desc_layout_);

    VkPipelineLayoutCreateInfo pl_info{};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &desc_layout_;
    vkCreatePipelineLayout(device_, &pl_info, nullptr, &pipeline_layout_);

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0, 0, static_cast<float>(width_), static_cast<float>(height_), 0, 1};
    VkRect2D sc{{0, 0}, {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)}};
    VkPipelineViewportStateCreateInfo vs{};
    vs.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vs.viewportCount = 1; vs.pViewports = &vp;
    vs.scissorCount = 1; vs.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    VkGraphicsPipelineCreateInfo gp_info{};
    gp_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp_info.stageCount = 2; gp_info.pStages = stages;
    gp_info.pVertexInputState = &vi;
    gp_info.pInputAssemblyState = &ia;
    gp_info.pViewportState = &vs;
    gp_info.pRasterizationState = &rs;
    gp_info.pMultisampleState = &ms;
    gp_info.pColorBlendState = &cb;
    gp_info.layout = pipeline_layout_;
    gp_info.renderPass = render_pass_;
    vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp_info, nullptr, &pipeline_);

    vkDestroyShaderModule(device_, vert_mod, nullptr);
    vkDestroyShaderModule(device_, frag_mod, nullptr);

    framebuffers_.resize(swapchain_views_.size());
    for (size_t i = 0; i < swapchain_views_.size(); ++i) {
        VkFramebufferCreateInfo fb_info{};
        fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass = render_pass_;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments = &swapchain_views_[i];
        fb_info.width = width_; fb_info.height = height_; fb_info.layers = 1;
        vkCreateFramebuffer(device_, &fb_info, nullptr, &framebuffers_[i]);
    }

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, RING_SIZE};
    VkDescriptorPoolCreateInfo dp_info{};
    dp_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_info.maxSets = RING_SIZE;
    dp_info.poolSizeCount = 1;
    dp_info.pPoolSizes = &ps;
    vkCreateDescriptorPool(device_, &dp_info, nullptr, &desc_pool_);

    std::vector<VkDescriptorSetLayout> layouts(RING_SIZE, desc_layout_);
    VkDescriptorSetAllocateInfo dsa_info{};
    dsa_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa_info.descriptorPool = desc_pool_;
    dsa_info.descriptorSetCount = RING_SIZE;
    dsa_info.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device_, &dsa_info, desc_sets_);

    for (size_t i = 0; i < RING_SIZE; ++i) {
        VkDescriptorImageInfo di_info{sampler_, gpu_image_views_[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet wds{};
        wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet = desc_sets_[i];
        wds.dstBinding = 0;
        wds.descriptorCount = 1;
        wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wds.pImageInfo = &di_info;
        vkUpdateDescriptorSets(device_, 1, &wds, 0, nullptr);
    }
}

void VulkanRenderer::create_sync_objects() {
    VkSemaphoreTypeCreateInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_info.initialValue = 0;

    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &timeline_info;
    vkCreateSemaphore(device_, &sem_info, nullptr, &timeline_semaphore_);

    VkSemaphoreCreateInfo bin_info{};
    bin_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (size_t i = 0; i < RING_SIZE; ++i) {
        vkCreateSemaphore(device_, &bin_info, nullptr, &image_available_sem_[i]);
        vkCreateSemaphore(device_, &bin_info, nullptr, &render_finished_sem_[i]);
    }
}

double VulkanRenderer::wait_slot_ready(size_t slot, uint64_t frame_idx) {
    (void)slot;
    if (frame_idx >= RING_SIZE) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t wait_gpu_val = (frame_idx - RING_SIZE) + 1;
        VkSemaphoreWaitInfo wait_info{};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &timeline_semaphore_;
        wait_info.pValues = &wait_gpu_val;
        vkWaitSemaphores(device_, &wait_info, UINT64_MAX);
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    return 0.0;
}

VulkanTimings VulkanRenderer::render_frame(size_t slot, uint64_t frame_idx) {
    VulkanTimings timings;
    uint64_t gpu_signal_val = frame_idx + 1;

    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t image_index = 0;
    VkResult acq_res = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                             image_available_sem_[slot], VK_NULL_HANDLE, &image_index);
    auto t1 = std::chrono::high_resolution_clock::now();
    timings.acquire_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    if (acq_res == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain();
        VkSemaphoreSignalInfo sig_info{};
        sig_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        sig_info.semaphore = timeline_semaphore_;
        sig_info.value = gpu_signal_val;
        vkSignalSemaphore(device_, &sig_info);
        return timings;
    }

    t0 = std::chrono::high_resolution_clock::now();
    VkCommandBuffer cmd = graphics_cmds_[slot];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi);

    VkImageMemoryBarrier ib{};
    ib.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = gpu_images_[slot];
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ib.srcAccessMask = 0;
    ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);

    VkBufferImageCopy copy_region{};
    copy_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.imageExtent = {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_), 1};
    vkCmdCopyBufferToImage(cmd, staging_buffers_[slot], gpu_images_[slot], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);

    VkRenderPassBeginInfo rp_bi{};
    rp_bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_bi.renderPass = render_pass_;
    rp_bi.framebuffer = framebuffers_[image_index];
    rp_bi.renderArea = {{0, 0}, {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)}};
    vkCmdBeginRenderPass(cmd, &rp_bi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &desc_sets_[slot], 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);

    vkEndCommandBuffer(cmd);
    t1 = std::chrono::high_resolution_clock::now();
    timings.record_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    VkSemaphore wait_sems[] = { image_available_sem_[slot] };
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

    VkSemaphore signal_sems[] = { render_finished_sem_[slot], timeline_semaphore_ };
    uint64_t signal_vals[] = { 0, gpu_signal_val };

    VkTimelineSemaphoreSubmitInfo timeline_submit{};
    timeline_submit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_submit.signalSemaphoreValueCount = 2;
    timeline_submit.pSignalSemaphoreValues = signal_vals;

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = &timeline_submit;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_sems;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    submit_info.signalSemaphoreCount = 2;
    submit_info.pSignalSemaphores = signal_sems;

    vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE);
    t1 = std::chrono::high_resolution_clock::now();
    timings.submit_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_finished_sem_[slot];
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_;
    present_info.pImageIndices = &image_index;

    vkQueuePresentKHR(graphics_queue_, &present_info);
    t1 = std::chrono::high_resolution_clock::now();
    timings.present_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    return timings;
}
