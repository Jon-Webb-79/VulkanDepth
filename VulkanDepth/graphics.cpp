// ===============================================================================
// ================================================================================
// - File:    graphics.cpp
// - Purpose: Describe the file purpose here
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    August 30, 2024
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here

#include "include/graphics.hpp"
#include "include/queues.hpp"
#include "include/stb_image.h"

#include <iostream>
#include <string>
// ================================================================================
// ================================================================================

DepthManager::DepthManager(AllocatorManager& allocatorManager, 
                           VkDevice device, 
                           VkPhysicalDevice physicalDevice, 
                           VkExtent2D swapChainExtent)
    : allocatorManager(allocatorManager),
      device(device),
      physicalDevice(physicalDevice),
      swapChainExtent(swapChainExtent) {
    createDepthResources();
}
// --------------------------------------------------------------------------------

DepthManager::~DepthManager() {
    cleanup();
}
// --------------------------------------------------------------------------------

void DepthManager::createDepthResources() {
    VkFormat depthFormat = findDepthFormat();

    // Create depth image and verify initialization
    createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, 
                VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 
                VMA_MEMORY_USAGE_GPU_ONLY, depthImage, depthImageMemory);
    
    // Check if depthImage and depthImageMemory are valid
    if (depthImage == VK_NULL_HANDLE || depthImageMemory == VK_NULL_HANDLE) {
        throw std::runtime_error("DepthManager: Failed to initialize depth image or memory allocation.");
    }

    // Create depth image view and verify initialization
    depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

    // Check if depthImageView is valid
    if (depthImageView == VK_NULL_HANDLE) {
        throw std::runtime_error("DepthManager: Failed to initialize depth image view.");
    }
}
// --------------------------------------------------------------------------------

void DepthManager::recreateDepthResources(VkExtent2D newExtent) {
    // Clean up existing resources
    cleanup();

    // Update swap chain extent and create new resources
    swapChainExtent = newExtent;
    createDepthResources();
}
// --------------------------------------------------------------------------------

VkFormat DepthManager::findDepthFormat() {
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
}
// --------------------------------------------------------------------------------

VkImageView DepthManager::getDepthImageView() {
    return depthImageView;
}
// ================================================================================

bool DepthManager::hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}
// --------------------------------------------------------------------------------

VkFormat DepthManager::findSupportedFormat(const std::vector<VkFormat>& candidates, 
                                           VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}
// --------------------------------------------------------------------------------

void DepthManager::createImage(uint32_t width, uint32_t height, VkFormat format, 
                               VkImageTiling tiling, VkImageUsageFlags usage, 
                               VmaMemoryUsage memoryUsage, VkImage& image, 
                               VmaAllocation& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memoryUsage;

    if (vmaCreateImage(allocatorManager.getAllocator(), &imageInfo, &allocInfo, &image, &imageMemory, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }
}
// --------------------------------------------------------------------------------

VkImageView DepthManager::createImageView(VkImage image, VkFormat format, 
                                          VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}
// --------------------------------------------------------------------------------

void DepthManager::cleanup() {
    if (depthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, depthImageView, nullptr);
        depthImageView = VK_NULL_HANDLE;
    }

    if (depthImage != VK_NULL_HANDLE && depthImageMemory != VK_NULL_HANDLE) {
        vmaDestroyImage(allocatorManager.getAllocator(), depthImage, depthImageMemory);
        depthImage = VK_NULL_HANDLE;
        depthImageMemory = VK_NULL_HANDLE;
    }
}
// ================================================================================
// ================================================================================

SamplerManager::SamplerManager(VkDevice device, VkPhysicalDevice physicalDevice)
    : device(device), physicalDevice(physicalDevice) {}
// --------------------------------------------------------------------------------

SamplerManager::~SamplerManager() {
    std::lock_guard<std::mutex> lock(samplerMutex);
    for (auto& sampler : samplers) {
        vkDestroySampler(device, sampler.second, nullptr);
    }
}
// --------------------------------------------------------------------------------

VkSampler SamplerManager::getSampler(const std::string& samplerKey) const {
    std::lock_guard<std::mutex> lock(samplerMutex);
    auto it = samplers.find(samplerKey);
    if (it != samplers.end()) {
        return it->second;
    } else {
        throw std::runtime_error("Sampler not found: " + samplerKey);
    }
}
// --------------------------------------------------------------------------------

VkSampler SamplerManager::createSampler(const std::string& samplerKey) {
    std::lock_guard<std::mutex> lock(samplerMutex);
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    VkSampler sampler;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }

    samplers[samplerKey] = sampler;
    return sampler;
}
// ================================================================================
// ================================================================================ 

DescriptorManager::DescriptorManager(VkDevice device)
    : device(device){
    createDescriptorSetLayout();
    createDescriptorPool();
}
// --------------------------------------------------------------------------------

    DescriptorManager::~DescriptorManager() {
    // Check and clean up descriptor set layout
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;  // Reset to null handle after destruction
    }

    // Check and clean up descriptor pool (this will automatically free descriptor sets)
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;  // Reset to null handle after destruction
    }

    // Clear the descriptor sets vector to free any associated memory
    descriptorSets.clear();
}
// --------------------------------------------------------------------------------

void DescriptorManager::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)}
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }
}
// --------------------------------------------------------------------------------

void DescriptorManager::createDescriptorSets(const std::vector<VkBuffer> uniformBuffers, 
                                             VkImageView textureImageView, 
                                             VkSampler textureSampler) {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
// --------------------------------------------------------------------------------

const VkDescriptorSetLayout& DescriptorManager::getDescriptorSetLayout() const{
    if (descriptorSetLayout == VK_NULL_HANDLE)
        throw std::runtime_error("Descriptor set layout is not initialized!");
    return descriptorSetLayout;
}
// --------------------------------------------------------------------------------

const VkDescriptorPool& DescriptorManager::getDescriptorPool() const {
    if (descriptorPool == VK_NULL_HANDLE)
        throw std::runtime_error("Descriptor pool is not initialized!");
    return descriptorPool;
}
// --------------------------------------------------------------------------------

const std::vector<VkDescriptorSet>& DescriptorManager::getDescriptorSets() const {
    if (descriptorSets.empty())
        throw std::runtime_error("Descriptor sets vector is empty!");
    return descriptorSets;
}
// --------------------------------------------------------------------------------

const VkDescriptorSet& DescriptorManager::getDescriptorSet(uint32_t frameIndex) const {
    // Bounds checking to prevent out-of-range access
    if (frameIndex >= descriptorSets.size()) {
        throw std::out_of_range("Frame index is out of bounds!");
    }
    
    if (descriptorSets[frameIndex] == VK_NULL_HANDLE) {
        throw std::runtime_error("Descriptor set for frame " + std::to_string(frameIndex) + " is not initialized!");
    }

    return descriptorSets[frameIndex];
}
// ================================================================================

void DescriptorManager::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}
// ================================================================================
// ================================================================================
// eof
