// ================================================================================
// ================================================================================
// - File:    application.cpp
// - Purpose: Describe the file purpose here
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    June 19, 2024
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here
#include "include/application.hpp"
#include "include/constants.hpp"

#include <vector>
#include <iostream>
// ================================================================================
// ================================================================================

VulkanInstance::VulkanInstance(GLFWwindow* window, ValidationLayers& validationLayers)
    : windowInstance(window), validationLayers(validationLayers) {

    createInstance();
    createSurface();
}
// --------------------------------------------------------------------------------

VulkanInstance::~VulkanInstance() {
    {
        std::lock_guard<std::mutex> lockSurface(surfaceMutex);
        if (surface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance, surface, nullptr);
            surface = VK_NULL_HANDLE;
        }
    }

    {
        std::lock_guard<std::mutex> lockInstance(instanceMutex);
        if (instance != VK_NULL_HANDLE) {
            validationLayers.cleanup(instance);
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }
}
// --------------------------------------------------------------------------------


VkInstance* VulkanInstance::getInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex);
    return &instance;
}
// --------------------------------------------------------------------------------

VkSurfaceKHR VulkanInstance::getSurface() {
    std::lock_guard<std::mutex> lock(surfaceMutex);
    return surface;
}
// ================================================================================

void VulkanInstance::createInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex);  // Protect creation of instance

    if (validationLayers.isEnabled() && !validationLayers.checkValidationLayerSupport()) {
        throw std::runtime_error("Validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanTriangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    uint32_t extensionCount = 0;
    const char** extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    std::vector<const char*> extensionVector(extensions, extensions + extensionCount);

    if (validationLayers.isEnabled()) {
        std::vector<const char*> validationLayerExtensions = validationLayers.getRequiredExtensions();
        extensionVector.insert(extensionVector.end(), validationLayerExtensions.begin(), validationLayerExtensions.end());
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensionVector.size());
    createInfo.ppEnabledExtensionNames = extensionVector.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (validationLayers.isEnabled()) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.getValidationLayers().size());
        createInfo.ppEnabledLayerNames = validationLayers.getValidationLayers().data();
        validationLayers.populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = &debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.ppEnabledLayerNames = nullptr;
        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }

    if (validationLayers.isEnabled()) {
        validationLayers.setupDebugMessenger(instance);
    }
}
// --------------------------------------------------------------------------------

void VulkanInstance::createSurface() {
    std::lock_guard<std::mutex> lock(surfaceMutex);
    if (glfwCreateWindowSurface(instance, windowInstance, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface\n");
}
// ================================================================================
// ================================================================================
// eof
