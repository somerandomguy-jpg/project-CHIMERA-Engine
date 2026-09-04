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

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "system2_coprocessor_suite.hpp"

#ifndef VK_LAYER_EXPORT
#define VK_LAYER_EXPORT __attribute__((visibility("default")))
#endif

namespace {

// Device dispatch functions
struct DeviceDispatchTable {
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr{nullptr};
    PFN_vkDestroyDevice DestroyDevice{nullptr};
    PFN_vkQueuePresentKHR QueuePresentKHR{nullptr};
};

// Instance dispatch functions
struct InstanceDispatchTable {
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr{nullptr};
    PFN_vkDestroyInstance DestroyInstance{nullptr};
};

struct LayerDeviceData {
    VkDevice device{VK_NULL_HANDLE};
    DeviceDispatchTable dispatch_table{};
    std::unique_ptr<avx512::CoprocessorEngine> engine;
    void* mapped_ssbo_ptr{nullptr};
};

std::mutex g_layer_lock;
std::unordered_map<void*, LayerDeviceData> g_device_map;
std::unordered_map<void*, InstanceDispatchTable> g_instance_map;

template <typename T>
void* get_dispatch_key(T obj) {
    return *reinterpret_cast<void**>(obj);
}

} // namespace

// ================================================================================================
// INSTANCE INTERCEPTIONS (vkCreateInstance & vkDestroyInstance)
// ================================================================================================

VKAPI_ATTR VkResult VKAPI_CALL AVX512_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance)
{
    VkLayerInstanceCreateInfo* chain_info = (VkLayerInstanceCreateInfo*)pCreateInfo->pNext;
    while (chain_info && (chain_info->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
                          chain_info->function != VK_LAYER_LINK_INFO)) {
        chain_info = (VkLayerInstanceCreateInfo*)chain_info->pNext;
    }

    if (!chain_info) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr gpa = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    PFN_vkCreateInstance create_fn = (PFN_vkCreateInstance)gpa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create_fn) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult result = create_fn(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS) return result;

    std::lock_guard<std::mutex> lock(g_layer_lock);
    InstanceDispatchTable table;
    table.GetInstanceProcAddr = gpa;
    table.DestroyInstance = (PFN_vkDestroyInstance)gpa(*pInstance, "vkDestroyInstance");
    g_instance_map[get_dispatch_key(*pInstance)] = table;

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL AVX512_DestroyInstance(
    VkInstance instance,
    const VkAllocationCallbacks* pAllocator)
{
    std::lock_guard<std::mutex> lock(g_layer_lock);
    void* key = get_dispatch_key(instance);
    auto it = g_instance_map.find(key);
    if (it != g_instance_map.end()) {
        if (it->second.DestroyInstance) {
            it->second.DestroyInstance(instance, pAllocator);
        }
        g_instance_map.erase(it);
    }
}

// ================================================================================================
// DEVICE INTERCEPTIONS (vkCreateDevice, vkDestroyDevice, vkQueuePresentKHR)
// ================================================================================================

VKAPI_ATTR VkResult VKAPI_CALL AVX512_CreateDevice(
    VkPhysicalDevice gpu,
    const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDevice* pDevice)
{
    VkLayerDeviceCreateInfo* chain_info = (VkLayerDeviceCreateInfo*)pCreateInfo->pNext;
    while (chain_info && (chain_info->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
                          chain_info->function != VK_LAYER_LINK_INFO)) {
        chain_info = (VkLayerDeviceCreateInfo*)chain_info->pNext;
    }

    if (!chain_info) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr gpa = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    PFN_vkCreateDevice create_fn = (PFN_vkCreateDevice)gpa(VK_NULL_HANDLE, "vkCreateDevice");
    if (!create_fn) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = create_fn(gpu, pCreateInfo, pAllocator, pDevice);
    if (res != VK_SUCCESS) return res;

    std::lock_guard<std::mutex> lock(g_layer_lock);
    LayerDeviceData data;
    data.device = *pDevice;
    data.dispatch_table.GetDeviceProcAddr = gdpa;
    data.dispatch_table.DestroyDevice = (PFN_vkDestroyDevice)gdpa(*pDevice, "vkDestroyDevice");
    data.dispatch_table.QueuePresentKHR = (PFN_vkQueuePresentKHR)gdpa(*pDevice, "vkQueuePresentKHR");
    data.engine = std::make_unique<avx512::CoprocessorEngine>();

    g_device_map[get_dispatch_key(*pDevice)] = std::move(data);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL AVX512_DestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks* pAllocator)
{
    std::lock_guard<std::mutex> lock(g_layer_lock);
    void* key = get_dispatch_key(device);
    auto it = g_device_map.find(key);
    if (it != g_device_map.end()) {
        if (it->second.dispatch_table.DestroyDevice) {
            it->second.dispatch_table.DestroyDevice(device, pAllocator);
        }
        g_device_map.erase(it);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL AVX512_QueuePresentKHR(
    VkQueue queue,
    const VkPresentInfoKHR* pPresentInfo)
{
    LayerDeviceData* dev_data = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_layer_lock);
        for (auto& pair : g_device_map) {
            dev_data = &pair.second;
            break;
        }
    }

    if (dev_data && dev_data->engine) {
        alignas(16) avx512::PointLightViewSpace dummy_lights[1];
        dummy_lights[0] = {0.0f, 0.0f, 5.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f};

        avx512::HDRPixelBuffer dummy_hdr;
        dummy_hdr.width = 1920;
        dummy_hdr.height = 1080;
        dummy_hdr.r.resize(1920 * 1080, 0.5f);
        dummy_hdr.g.resize(1920 * 1080, 0.5f);
        dummy_hdr.b.resize(1920 * 1080, 0.5f);
        std::vector<uint32_t> dummy_ldr(1920 * 1080);

        dev_data->engine->execute_frame(
            dummy_lights, 1,
            nullptr, nullptr,
            dummy_hdr, dummy_ldr.data(),
            1920, 1080
        );

        if (dev_data->mapped_ssbo_ptr) {
            std::memcpy(dev_data->mapped_ssbo_ptr,
                        &dev_data->engine->get_light_table(),
                        sizeof(avx512::ClusterLightBitmaskTable));
        }
    }

    if (dev_data && dev_data->dispatch_table.QueuePresentKHR) {
        return dev_data->dispatch_table.QueuePresentKHR(queue, pPresentInfo);
    }
    return VK_SUCCESS;
}

// ================================================================================================
// EXPORTED GET_PROC_ADDR ENTRYPOINTS
// ================================================================================================

extern "C" {

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL AVX512_GetDeviceProcAddr(
    VkDevice device,
    const char* pName)
{
    if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)AVX512_GetDeviceProcAddr;
    if (std::strcmp(pName, "vkDestroyDevice") == 0)      return (PFN_vkVoidFunction)AVX512_DestroyDevice;
    if (std::strcmp(pName, "vkQueuePresentKHR") == 0)    return (PFN_vkVoidFunction)AVX512_QueuePresentKHR;

    if (device != VK_NULL_HANDLE) {
        std::lock_guard<std::mutex> lock(g_layer_lock);
        auto it = g_device_map.find(get_dispatch_key(device));
        if (it != g_device_map.end() && it->second.dispatch_table.GetDeviceProcAddr) {
            return it->second.dispatch_table.GetDeviceProcAddr(device, pName);
        }
    }
    return nullptr;
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL AVX512_GetInstanceProcAddr(
    VkInstance instance,
    const char* pName)
{
    if (std::strcmp(pName, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)AVX512_GetInstanceProcAddr;
    if (std::strcmp(pName, "vkCreateInstance") == 0)      return (PFN_vkVoidFunction)AVX512_CreateInstance;
    if (std::strcmp(pName, "vkDestroyInstance") == 0)     return (PFN_vkVoidFunction)AVX512_DestroyInstance;
    if (std::strcmp(pName, "vkCreateDevice") == 0)        return (PFN_vkVoidFunction)AVX512_CreateDevice;
    if (std::strcmp(pName, "vkDestroyDevice") == 0)       return (PFN_vkVoidFunction)AVX512_DestroyDevice;
    if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0)   return (PFN_vkVoidFunction)AVX512_GetDeviceProcAddr;
    if (std::strcmp(pName, "vkQueuePresentKHR") == 0)     return (PFN_vkVoidFunction)AVX512_QueuePresentKHR;

    if (instance != VK_NULL_HANDLE) {
        std::lock_guard<std::mutex> lock(g_layer_lock);
        auto it = g_instance_map.find(get_dispatch_key(instance));
        if (it != g_instance_map.end() && it->second.GetInstanceProcAddr) {
            return it->second.GetInstanceProcAddr(instance, pName);
        }
    }
    return nullptr;
}

// Standard fallback aliases
VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    return AVX512_GetInstanceProcAddr(instance, pName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    return AVX512_GetDeviceProcAddr(device, pName);
}

// Interface version negotiation
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(
    VkNegotiateLayerInterface* pVersionStruct)
{
    if (!pVersionStruct || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->pfnGetInstanceProcAddr = AVX512_GetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = AVX512_GetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    }
    return VK_SUCCESS;
}

} // extern "C"
