// ================================================================================
// ================================================================================
// - File:    application.hpp
// - Purpose: This file contains a basic application interface for building and 
//            displaying a triangle to the screen.
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    June 19, 2024
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================

#ifndef application_HPP
#define application_HPP

// Define Preprocessor Macros (before including related libraries)
#ifndef GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_VULKAN
#endif

#ifndef GLM_FORCE_RADIANS 
#define GLM_FORCE_RADIANS
#endif 

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE 
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif

// External Libraries
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

// Standard Library Includes
#include <memory>
#include <mutex>
#include <string>
#include <chrono>

// Project-Specific Headers
#include "validation_layers.hpp"
#include "memory.hpp"
#include "constants.hpp"
#include "graphics.hpp"
#include "devices.hpp"
// ================================================================================
// ================================================================================

/**
 * @brief This class creates an instance of Vulkan to support an 
 * application that will draw a triangle to the screen
 */
class VulkanInstance {
public:

    /**
     * @brief Constructor for the VulkanInstance class 
     *
     * @param window A reference to a Window object
     */
    VulkanInstance(GLFWwindow* window, ValidationLayers& validationLayers);
// --------------------------------------------------------------------------------

    /**
     * @brief Destructor for the VulkanInstance class
     */
    ~VulkanInstance();
// --------------------------------------------------------------------------------

    /**
     * @brief Returns a raw pointer to the instance of Vulkan
     */
    VkInstance* getInstance();
// --------------------------------------------------------------------------------

    /**
     * @brief Returns a raw pointer to an instance of surface
     */
    VkSurfaceKHR getSurface();
// ================================================================================
private:
    GLFWwindow* windowInstance;
    ValidationLayers& validationLayers;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    std::mutex instanceMutex;
    std::mutex surfaceMutex;
// --------------------------------------------------------------------------------

    /**
     * @brief Helper function that allows the constructor to create a Vulkan instance
     */
    void createInstance();
// --------------------------------------------------------------------------------

    /**
     * @brief Helper function that establishes a relationship between Vulkan and the window
     */
    void createSurface();
};
// ================================================================================ 
// ================================================================================

/**
 * @brief A template class that represents the Vulkan application.
 * 
 * This class encapsulates the core components and workflow of a Vulkan-based rendering application.
 * It supports customizable vertex and index types for flexibility in handling various mesh structures.
 * The application manages window events, rendering loop, and Vulkan resource setup and cleanup.
 * 
 * @tparam VertexType The type of vertex data, typically used for position, color, and texture coordinates.
 * @tparam IndexType The type of index data, used for indexed drawing; must be uint8_t, uint16_t, or uint32_t.
 */
template <typename VertexType, typename IndexType>
class VulkanApplication {
public:
    float zoomLevel = 1.0f;
// --------------------------------------------------------------------------------

    /**
     * @brief Ensures IndexType is a valid Vulkan-compatible type.
     * 
     * This static assertion verifies that IndexType is either uint8_t, uint16_t, or uint32_t.
     * Other types will result in a compile-time error, as Vulkan only supports these index formats.
     */
    static_assert(std::is_same<IndexType, uint16_t>::value || 
                  std::is_same<IndexType, uint32_t>::value || 
                  std::is_same<IndexType, uint8_t>::value, 
                  "IndexType must be uint16_t, uint32_t, or uint8_t for Vulkan.");
// --------------------------------------------------------------------------------

    /**
     * @brief Constructs a VulkanApplication with a specified window and mesh data.
     * 
     * Initializes the application using the provided window, vertices, and indices.
     * It also sets up core Vulkan resources, such as the pipeline and command buffers.
     * 
     * @param window A pointer to the GLFWwindow instance.
     * @param vertices A vector of vertices of type VertexType.
     * @param indices A vector of indices of type IndexType.
     */
    VulkanApplication(GLFWwindow* window, 
                      const std::vector<VertexType>& vertices,
                      const std::vector<IndexType>& indices)
        : windowInstance(std::move(window)),
          vertices(vertices),
          indices(indices){
        std::string samplerString = "default";
        std::string vertexString = "../../shaders/shader.vert.spv";
        std::string fragString = "../../shaders/shader.frag.spv";
        constructor(samplerString, vertexString, fragString);
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Destructor for VulkanApplication.
     * 
     * Cleans up all Vulkan resources by calling destroyResources().
     */
    ~VulkanApplication() {
        destroyResources();
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Callback function for scroll events to adjust zoom level.
     * 
     * Modifies the zoom level based on scroll input, clamping the zoom level to a
     * defined range to avoid excessive zooming.
     * 
     * @param window The GLFW window receiving the scroll event.
     * @param xoffset The horizontal scroll offset.
     * @param yoffset The vertical scroll offset, used to modify zoom.
     */
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        VulkanApplication* app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(window));
        if (app) {
            app->zoomLevel -= yoffset * 0.1f; // Adjust zoom sensitivity
            app->zoomLevel = glm::clamp(app->zoomLevel, 0.1f, 5.0f); // Clamp zoom level to reasonable limits
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Runs the main application loop.
     * 
     * Polls window events, renders frames, and handles swap chain recreation when necessary.
     * The loop continues until the window is closed.
     */
    void run() {
        glfwSetScrollCallback(windowInstance, scrollCallback);
        while (!glfwWindowShouldClose(windowInstance)) {
            glfwPollEvents();
            drawFrame();

            if (framebufferResized) {
                recreateSwapChain();
                framebufferResized = false;
            }
        }
        vkDeviceWaitIdle(vulkanLogicalDevice->getDevice());
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Sets the framebuffer resize status.
     * 
     * Updates the framebufferResized flag, which triggers swap chain recreation in the
     * main loop if set to true.
     * 
     * @param resized A boolean indicating if the framebuffer has been resized.
     */
    void setFramebufferResized(bool resized) { framebufferResized = resized; }
// ================================================================================ 
private:

    GLFWwindow* windowInstance; /**< The GLFW window associated with this application. */ 
    std::unique_ptr<ValidationLayers> validationLayers; /**< Handles Vulkan validation layers. */ 
    std::unique_ptr<VulkanInstance> vulkanInstanceCreator; /**< Creates the Vulkan instance. */ 
    std::unique_ptr<VulkanPhysicalDevice> vulkanPhysicalDevice; /**< Manages the Vulkan physical device. */ 
    std::unique_ptr<VulkanLogicalDevice> vulkanLogicalDevice; /**< Handles the Vulkan logical device. */ 
    std::unique_ptr<SwapChain> swapChain; /**< Manages swap chain creation and resizing. */ 
    std::unique_ptr<DepthManager> depthManager; /**< Manages depth buffering resources. */ 
    std::unique_ptr<CommandBufferManager<IndexType>> commandBufferManager; /**< Manages Vulkan command buffers. */ 
    std::unique_ptr<SamplerManager> samplerManager; /**< Handles Vulkan texture samplers. */ 
    std::unique_ptr<TextureManager<IndexType>> textureManager; /**< Manages Vulkan textures. */ 
    std::unique_ptr<BufferManager<VertexType, IndexType>> bufferManager; /**< Manages Vulkan buffers for vertex and index data. */ 
    std::unique_ptr<DescriptorManager> descriptorManager; /**< Manages descriptor sets for shaders. */ 
    std::unique_ptr<GraphicsPipeline<VertexType, IndexType>> graphicsPipeline; /**< Manages the Vulkan graphics pipeline. */ 

    std::vector<VertexType> vertices; /**< A vector holding vertex data. */ 
    std::vector<IndexType> indices; /**< A vector holding index data. */ 
    VkQueue graphicsQueue; /**< Vulkan queue for graphics operations. */ 
    VkQueue presentQueue; /**< Vulkan queue for presentation operations. */ 

    std::unique_ptr<AllocatorManager> allocatorManager; /**< Handles memory allocation for Vulkan objects. */ 
    uint32_t currentFrame = 0; /**< Tracks the current frame index for rendering synchronization. */ 
    bool framebufferResized = false; /**< Indicates if the framebuffer has been resized. */ 
// --------------------------------------------------------------------------------

    /**
     * @brief Helper method to initialize core Vulkan components.
     * 
     * Sets up Vulkan resources, including command buffers, depth buffers, and pipeline objects.
     * This method is called in the constructor.
     * 
     * @param samplerString The name of the texture sampler.
     * @param vertexString The file path to the vertex shader.
     * @param fragString The file path to the fragment shader.
     */
    void constructor(std::string samplerstring, std::string vertexstring, std::string fragstring) {
        glfwSetWindowUserPointer(windowInstance, this);

        validationLayers = std::make_unique<ValidationLayers>();
        vulkanInstanceCreator = std::make_unique<VulkanInstance>(this->windowInstance, 
                                                                 *validationLayers.get());
        vulkanPhysicalDevice = std::make_unique<VulkanPhysicalDevice>(*this->vulkanInstanceCreator->getInstance(),
                                                                      this->vulkanInstanceCreator->getSurface());
        vulkanLogicalDevice = std::make_unique<VulkanLogicalDevice>(vulkanPhysicalDevice->getDevice(),
                                                                    validationLayers->getValidationLayers(),
                                                                    vulkanInstanceCreator->getSurface(),
                                                                    deviceExtensions);
        allocatorManager = std::make_unique<AllocatorManager>(
            vulkanPhysicalDevice->getDevice(),
            vulkanLogicalDevice->getDevice(),
            *vulkanInstanceCreator->getInstance());

        swapChain = std::make_unique<SwapChain>(vulkanLogicalDevice->getDevice(),
                                                vulkanInstanceCreator->getSurface(),
                                                vulkanPhysicalDevice->getDevice(),
                                                this->windowInstance);
        depthManager = std::make_unique<DepthManager>(
            *allocatorManager,
            vulkanLogicalDevice->getDevice(),
            vulkanPhysicalDevice->getDevice(),
            swapChain->getSwapChainExtent()
        );
        commandBufferManager = std::make_unique<CommandBufferManager<IndexType>>(vulkanLogicalDevice->getDevice(),
                                                                      indices,
                                                                      vulkanPhysicalDevice->getDevice(),
                                                                      vulkanInstanceCreator->getSurface());
        samplerManager = std::make_unique<SamplerManager>(
                vulkanLogicalDevice->getDevice(),
                vulkanPhysicalDevice->getDevice()
        );
        samplerManager->createSampler(samplerstring);
        textureManager = std::make_unique<TextureManager<IndexType>>(
            *allocatorManager,
            vulkanLogicalDevice->getDevice(),
            vulkanPhysicalDevice->getDevice(),
            *commandBufferManager,     
            vulkanLogicalDevice->getGraphicsQueue(),
            "../../../data/texture.jpg",
            *samplerManager,
            samplerstring
        );
        bufferManager = std::make_unique<BufferManager<VertexType, IndexType>>(vertices,
                                                        indices,
                                                        *allocatorManager,
                                                        *commandBufferManager.get(),
                                                        vulkanLogicalDevice->getGraphicsQueue());
        descriptorManager = std::make_unique<DescriptorManager>(vulkanLogicalDevice->getDevice());
        descriptorManager->createDescriptorSets(bufferManager->getUniformBuffers(),
                                                textureManager->getTextureImageView(),
                                                samplerManager->getSampler(samplerstring)
                                                );
        // graphicsPipeline = std::make_unique<GraphicsPipelineTwo<VertexType, IndexType>>(vulkanLogicalDevice->getDevice(),
        //                                                       *swapChain.get());
        graphicsPipeline = std::make_unique<GraphicsPipeline<VertexType, IndexType>>(vulkanLogicalDevice->getDevice(),
                                                              *swapChain.get(),
                                                              *commandBufferManager.get(),
                                                              *bufferManager.get(),
                                                              *descriptorManager.get(),
                                                              indices,
                                                              *vulkanPhysicalDevice.get(),
                                                              vertexstring,
                                                              fragstring,
                                                              *depthManager.get());
        graphicsPipeline->createFrameBuffers(swapChain->getSwapChainImageViews(), 
                                             swapChain->getSwapChainExtent());
        graphicsQueue = this->vulkanLogicalDevice->getGraphicsQueue();
        presentQueue = this->vulkanLogicalDevice->getPresentQueue();
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Cleans up Vulkan resources.
     * 
     * Resets all smart pointers and Vulkan handles to free memory and prevent resource leaks.
     * Called in the destructor.
     */
    void destroyResources() {
        graphicsPipeline.reset();
        descriptorManager.reset();
        commandBufferManager.reset();
     
        samplerManager.reset();
        textureManager.reset();
        bufferManager.reset(); 
        depthManager.reset();
        swapChain.reset();
     
        // Reset allocator before logical device
        allocatorManager.reset();

        // Destroy Vulkan logical device
        vulkanLogicalDevice.reset();

        // Destroy other Vulkan resources
        vulkanPhysicalDevice.reset();
        vulkanInstanceCreator.reset();
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Renders a single frame.
     * 
     * This method acquires a swap chain image, records commands, submits them to the graphics queue,
     * and presents the image. It handles synchronization and swap chain recreation when necessary.
     * 
     * @throws std::runtime_error if Vulkan function calls fail during rendering.
     */
    void drawFrame() {
        VkDevice device = vulkanLogicalDevice->getDevice();
        uint32_t frameIndex = currentFrame;

        // Wait for the frame to be finished
        commandBufferManager->waitForFences(frameIndex);
        commandBufferManager->resetFences(frameIndex);
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain->getSwapChain(), UINT64_MAX, 
                                                commandBufferManager->getImageAvailableSemaphore(frameIndex), 
                                                VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain(); // Recreate swap chain if it's out of date
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Update the uniform buffer with the current image/frame
        updateUniformBuffer(frameIndex);

        VkCommandBuffer cmdBuffer = commandBufferManager->getCommandBuffer(frameIndex);

        vkResetCommandBuffer(cmdBuffer, 0);

        graphicsPipeline->recordCommandBuffer(frameIndex, imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {commandBufferManager->getImageAvailableSemaphore(frameIndex)};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        VkSemaphore signalSemaphores[] = {commandBufferManager->getRenderFinishedSemaphore(frameIndex)};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, commandBufferManager->getInFlightFence(frameIndex)) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain->getSwapChain()};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();  // Recreate swap chain if it's out of date or suboptimal
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Recreates the swap chain and dependent resources.
     * 
     * Called when the framebuffer is resized. Reinitializes swap chain resources and command buffers
     * to accommodate the new window size.
     */
    void recreateSwapChain() {
        // If the window is minimized, pause execution until the window is resized again
        int width = 0, height = 0;
        GLFWwindow* glfwWindow = static_cast<GLFWwindow*>(windowInstance);
        glfwGetFramebufferSize(glfwWindow, &width, &height);

        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(glfwWindow, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(vulkanLogicalDevice->getDevice());

        // Clean up existing swap chain and related resources
        graphicsPipeline->destroyFramebuffers();
        swapChain->cleanupSwapChain();

        // Recreate the swap chain and dependent resources
        swapChain->recreateSwapChain();

        // Recreate depth resources with the new extent
        depthManager->recreateDepthResources(swapChain->getSwapChainExtent());

        // Recreate the framebuffers using the new swap chain image views
        graphicsPipeline->createFrameBuffers(swapChain->getSwapChainImageViews(), swapChain->getSwapChainExtent());

        // Free existing command buffers
        VkCommandPool commandPool = commandBufferManager->getCommandPool();
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkCommandBuffer cmdBuffer = commandBufferManager->getCommandBuffer(i);
            vkFreeCommandBuffers(vulkanLogicalDevice->getDevice(), commandPool, 1, &cmdBuffer);
        }

        // Recreate the command buffers
        commandBufferManager->createCommandBuffers();
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Callback for framebuffer resize events.
     * 
     * Triggers a flag for swap chain recreation when the window framebuffer is resized.
     * 
     * @param window The GLFW window receiving the resize event.
     * @param width The new framebuffer width.
     * @param height The new framebuffer height.
     */
    void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(window));
        if (app) {
            app->setFramebufferResized(true);
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Updates the uniform buffer for a given frame.
     * 
     * Recalculates the transformation matrices for model, view, and projection
     * and uploads them to the uniform buffer for the specified frame.
     * 
     * @param currentImage The index of the current swap chain image.
     */
    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        float fov = glm::radians(45.0f) / zoomLevel; // Adjust FOV with zoom level
        ubo.proj = glm::perspective(fov, swapChain->getSwapChainExtent().width / (float)swapChain->getSwapChainExtent().height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1; // Invert Y-axis for Vulkan

        memcpy(bufferManager->getUniformBuffersMapped()[currentImage], &ubo, sizeof(ubo));
    }
};
// ================================================================================
// ================================================================================
#endif /* application_HPP */
// eof
