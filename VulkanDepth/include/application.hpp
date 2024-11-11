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

// Forward declaration to support VulkanApplicationBuilder
template <typename VertexType, typename IndexType>
class VulkanApplication;

/**
 * @brief A builder class for constructing VulkanApplication instances.
 * 
 * VulkanApplicationBuilder helps configure and instantiate a VulkanApplication
 * object with custom settings and resource paths. The builder pattern provides
 * a flexible, readable way to set up essential Vulkan resources, shaders, and 
 * textures, streamlining VulkanApplication initialization.
 *
 * @tparam VertexType The vertex data type to use in the application.
 * @tparam IndexType The index data type, restricted to uint8_t, uint16_t, or uint32_t, compatible with Vulkan.
 */
template <typename VertexType, typename IndexType>
class VulkanApplicationBuilder {
public:
    static_assert(std::is_same<IndexType, uint16_t>::value || 
                  std::is_same<IndexType, uint32_t>::value || 
                  std::is_same<IndexType, uint8_t>::value, 
                  "IndexType must be uint16_t, uint32_t, or uint8_t for Vulkan.");
// --------------------------------------------------------------------------------

    /**
     * @brief Sets the vertex and index data for the Vulkan application.
     * 
     * Specifies the vertex and index buffers to be used in the Vulkan application.
     * 
     * @param vertices A vector containing vertex data of type VertexType.
     * @param indices A vector containing index data of type IndexType.
     * @return A reference to this builder for chained calls.
     */
    VulkanApplicationBuilder& setVertexInfo(const std::vector<VertexType>& vertices,
                                            const std::vector<IndexType>& indices) {
        this->vertices = vertices;
        this->indices = indices;
        return *this;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Sets the file paths for vertex and fragment shader files.
     * 
     * Defines the SPIR-V shader file paths for the vertex and fragment shaders.
     * These files are necessary for creating the graphics pipeline in the application.
     * 
     * @param vertPath Path to the vertex shader SPIR-V file.
     * @param fragPath Path to the fragment shader SPIR-V file.
     * @return A reference to this builder for chained calls.
     */
    VulkanApplicationBuilder& setBaseShaderPaths(const std::string& vertPath, 
                                                 const std::string& fragPath) {
       this->vertPath = vertPath;
       this->fragPath = fragPath;
       return *this;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Sets the file path for the compute shader.
     * 
     * Defines the SPIR-V shader file path for the compute shader, if used.
     * This shader path is optional and only needed if compute shaders are part of the application.
     * 
     * @param computePath Path to the compute shader SPIR-V file.
     * @return A reference to this builder for chained calls.
     */
    VulkanApplicationBuilder& setComputerShaderPath(const std::string& computePath) {
        this->computePath = computePath;
        return *this;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Sets the file path for the texture image.
     * 
     * Defines the file path to the texture image to be loaded and used in the application.
     * 
     * @param texturePath Path to the texture image file.
     * @return A reference to this builder for chained calls.
     */
    VulkanApplicationBuilder& setTexturePath(const std::string& texturePath) {
        this->texturePath = texturePath;
        return *this;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Sets the sampler configuration string.
     * 
     * Defines a string to identify or configure the sampler used for textures in the application.
     * This can be used to select or configure samplers if multiple sampler configurations are present.
     * 
     * @param samplerString A string identifier for the sampler.
     * @return A reference to this builder for chained calls.
     */
    VulkanApplicationBuilder& setSamplerString(const std::string& samplerString) {
        this->samplerString = samplerString;
        return *this;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Builds and returns a configured VulkanApplication instance.
     * 
     * Creates the VulkanApplication instance with the configurations specified in the builder,
     * including vertices, indices, shader paths, and texture information. Initializes necessary
     * Vulkan components such as swap chain, depth buffer, and command buffers.
     * 
     * @param screenWidth The width of the application window.
     * @param screenHeight The height of the application window.
     * @param title The title of the application window.
     * @param fullScreen A boolean to specify if the application should run in full-screen mode.
     * @return A VulkanApplication instance, fully configured with the specified settings.
     * 
     * @throws std::runtime_error if mandatory settings (like shader paths) are missing
     * or if any Vulkan resource creation fails.
     */
    VulkanApplication<VertexType, IndexType> build(uint32_t screenWidth, uint32_t screenHeight,
                                                   const std::string& title, bool fullScreen = false) {
        // Create GLFW Window 
        windowInstance = createWindow(screenHeight, screenWidth, title, fullScreen);

        if (vertPath.empty() || fragPath.empty()) {
            throw std::runtime_error("Frag and Vert Shader paths must be specified");
        }

        auto validationLayers = std::make_unique<ValidationLayers>();

        auto vulkanInstanceCreator = std::make_unique<VulkanInstance>(
            this->windowInstance, 
            *validationLayers.get()
        );

        auto vulkanPhysicalDevice = std::make_unique<VulkanPhysicalDevice>(
            *vulkanInstanceCreator->getInstance(),
            vulkanInstanceCreator->getSurface()
        );

        auto vulkanLogicalDevice = std::make_unique<VulkanLogicalDevice>(
            vulkanPhysicalDevice->getDevice(),
            validationLayers->getValidationLayers(),
            vulkanInstanceCreator->getSurface(),
            deviceExtensions
        );

        auto allocatorManager = std::make_unique<AllocatorManager>(
            vulkanPhysicalDevice->getDevice(),
            vulkanLogicalDevice->getDevice(),
            *vulkanInstanceCreator->getInstance()
        );

        auto swapChain = std::make_unique<SwapChain>(
            vulkanLogicalDevice->getDevice(),
            vulkanInstanceCreator->getSurface(),
            vulkanPhysicalDevice->getDevice(),
            this->windowInstance
        );

        auto depthManager = std::make_unique<DepthManager>(
            *allocatorManager,
            vulkanLogicalDevice->getDevice(),
            vulkanPhysicalDevice->getDevice(),
            swapChain->getSwapChainExtent()
        );

        auto commandBufferManager = std::make_unique<CommandBufferManager<IndexType>>(
            vulkanLogicalDevice->getDevice(),
            indices,
            vulkanPhysicalDevice->getDevice(),
            vulkanInstanceCreator->getSurface()
        );

        auto samplerManager = std::make_unique<SamplerManager>(
            vulkanLogicalDevice->getDevice(),
            vulkanPhysicalDevice->getDevice()
        );
        samplerManager->createSampler(samplerString);

        auto textureManager = std::make_unique<TextureManager<IndexType>>(
            *allocatorManager,
            vulkanLogicalDevice->getDevice(),
            vulkanPhysicalDevice->getDevice(),
            *commandBufferManager,     
            vulkanLogicalDevice->getGraphicsQueue(),
            texturePath,
            *samplerManager,
            samplerString
        );

        auto bufferManager = std::make_unique<BufferManager<VertexType, IndexType>>(
            vertices,
            indices,
            *allocatorManager.get(),
            *commandBufferManager.get(),
            vulkanLogicalDevice->getGraphicsQueue()
        );

        auto descriptorManager = std::make_unique<DescriptorManager>(vulkanLogicalDevice->getDevice());
        descriptorManager->createDescriptorSets(
            bufferManager->getUniformBuffers(),
            textureManager->getTextureImageView(),
            samplerManager->getSampler(samplerString)
        );

        auto graphicsPipeline = std::make_unique<GraphicsPipeline<VertexType, IndexType>>(
            vulkanLogicalDevice->getDevice(),
            *swapChain.get(),
            *commandBufferManager.get(),
            *bufferManager.get(),
            *descriptorManager.get(),
            indices,
            *vulkanPhysicalDevice.get(),
            vertPath,
            fragPath,
            *depthManager.get()
        );
        graphicsPipeline->createFrameBuffers(swapChain->getSwapChainImageViews(), 
                                             swapChain->getSwapChainExtent());

        VkQueue graphicsQueue = vulkanLogicalDevice->getGraphicsQueue();
        VkQueue presentQueue = vulkanLogicalDevice->getPresentQueue();

        return VulkanApplication(
            vertices,
            indices,
            graphicsQueue,
            presentQueue,
            windowInstance,
            std::move(validationLayers),
            std::move(vulkanInstanceCreator),
            std::move(vulkanPhysicalDevice),
            std::move(vulkanLogicalDevice),
            std::move(swapChain),
            std::move(depthManager),
            std::move(commandBufferManager),
            std::move(samplerManager),
            std::move(textureManager),
            std::move(bufferManager),
            std::move(descriptorManager),
            std::move(graphicsPipeline),
            std::move(allocatorManager)
        );
    }
// ================================================================================
private:
    GLFWwindow* windowInstance; /**< The GLFW window instance to be used by Vulkan. */
    std::vector<VertexType> vertices; /**< Vertex data used in the application. */
    std::vector<IndexType> indices; /**< Index data used in the application. */
    std::string vertPath; /**< Path to the vertex shader file. */
    std::string fragPath; /**< Path to the fragment shader file. */
    std::string computePath; /**< Path to the compute shader file, if used. */
    std::string texturePath; /**< Path to the texture image file. */
    std::string samplerString = "default"; /**< Sampler identifier string, defaulted to "default". */
// --------------------------------------------------------------------------------

    /**
     * @brief Creates a GLFW window for Vulkan rendering.
     * 
     * Sets up a GLFW window with specified dimensions, title, and full-screen setting.
     * Configures GLFW for Vulkan by specifying no client API and setting window resizing hints.
     * 
     * @param h The height of the window.
     * @param w The width of the window.
     * @param screen_title The title of the window.
     * @param full_screen Boolean to specify if the window should be in full-screen mode.
     * @return A pointer to the created GLFW window.
     * 
     * @throws std::runtime_error if GLFW initialization or window creation fails.
     */
    GLFWwindow* createWindow(uint32_t h, uint32_t w, const std::string& screen_title,
                             bool full_screen) {
        if (!glfwInit()) {
            throw std::runtime_error("GLFW Initialization Failed!\n");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        GLFWmonitor* monitor = full_screen ? glfwGetPrimaryMonitor() : nullptr;

        GLFWwindow* window = glfwCreateWindow(w, h, screen_title.c_str(), monitor, nullptr);

        if (!window) {
            glfwTerminate();
            throw std::runtime_error("GLFW Instantiation failed!\n");
        }
        return window;
    }
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
     * @brief Constructs a VulkanApplication with the specified parameters.
     * 
     * Initializes the Vulkan application with essential resources for rendering,
     * including command buffers, swap chains, texture managers, and graphics pipelines.
     * This constructor accepts a set of Vulkan-related managers and configurations
     * that must be initialized prior to constructing the VulkanApplication.
     * 
     * @param vertices A vector of VertexType containing the vertex data to be used in the application.
     * @param indices A vector of IndexType containing the index data for rendering.
     * @param graphicsQueue The VkQueue handle for submitting graphics operations.
     * @param presentQueue The VkQueue handle for presenting images to the screen.
     * @param windowInstance A pointer to the GLFW window instance used for rendering.
     * @param validationLayers A unique pointer to the ValidationLayers manager, responsible for enabling and configuring Vulkan validation layers.
     * @param vulkanInstanceCreator A unique pointer to the VulkanInstance manager, responsible for creating and managing the Vulkan instance.
     * @param vulkanPhysicalDevice A unique pointer to the VulkanPhysicalDevice manager, responsible for selecting and managing the physical device used for rendering.
     * @param vulkanLogicalDevice A unique pointer to the VulkanLogicalDevice manager, responsible for creating and managing the logical device.
     * @param swapChain A unique pointer to the SwapChain manager, responsible for managing image presentation and swap chain configuration.
     * @param depthManager A unique pointer to the DepthManager, which manages depth buffering resources for 3D rendering.
     * @param commandBufferManager A unique pointer to the CommandBufferManager, managing command buffer allocation, submission, and synchronization.
     * @param samplerManager A unique pointer to the SamplerManager, responsible for creating and managing texture samplers.
     * @param textureManager A unique pointer to the TextureManager, handling texture loading, memory allocation, and image views.
     * @param bufferManager A unique pointer to the BufferManager, managing vertex and index buffers.
     * @param descriptorManager A unique pointer to the DescriptorManager, responsible for creating and binding descriptor sets for shaders.
     * @param graphicsPipeline A unique pointer to the GraphicsPipeline, which configures and manages the rendering pipeline, including shader stages.
     * @param allocatorManager A unique pointer to the AllocatorManager, managing memory allocation for Vulkan resources.
     * 
     * @note This constructor sets the window's user pointer to the VulkanApplication instance,
     * enabling the use of GLFW window callbacks to directly reference the application instance.
     * 
     * @throws std::runtime_error if any Vulkan objects fail to initialize.
     */

    VulkanApplication(
        std::vector<VertexType>& vertices,
        std::vector<IndexType>& indices,
        VkQueue graphicsQueue,
        VkQueue presentQueue,
        GLFWwindow* windowInstance,
        std::unique_ptr<ValidationLayers> validationLayers,
        std::unique_ptr<VulkanInstance> vulkanInstanceCreator,
        std::unique_ptr<VulkanPhysicalDevice> vulkanPhysicalDevice,
        std::unique_ptr<VulkanLogicalDevice> vulkanLogicalDevice,
        std::unique_ptr<SwapChain> swapChain,
        std::unique_ptr<DepthManager> depthManager,
        std::unique_ptr<CommandBufferManager<IndexType>> commandBufferManager,
        std::unique_ptr<SamplerManager> samplerManager,
        std::unique_ptr<TextureManager<IndexType>> textureManager,
        std::unique_ptr<BufferManager<VertexType, IndexType>> bufferManager,
        std::unique_ptr<DescriptorManager> descriptorManager,
        std::unique_ptr<GraphicsPipeline<VertexType, IndexType>> graphicsPipeline,
        std::unique_ptr<AllocatorManager> allocatorManager
    )
    : vertices(vertices),
      indices(indices),
      graphicsQueue(graphicsQueue),
      presentQueue(presentQueue),
      windowInstance(windowInstance),
      validationLayers(std::move(validationLayers)),
      vulkanInstanceCreator(std::move(vulkanInstanceCreator)),
      vulkanPhysicalDevice(std::move(vulkanPhysicalDevice)),
      vulkanLogicalDevice(std::move(vulkanLogicalDevice)),
      swapChain(std::move(swapChain)),
      depthManager(std::move(depthManager)),
      commandBufferManager(std::move(commandBufferManager)),
      samplerManager(std::move(samplerManager)),
      textureManager(std::move(textureManager)),
      bufferManager(std::move(bufferManager)),
      descriptorManager(std::move(descriptorManager)),
      graphicsPipeline(std::move(graphicsPipeline)),
      allocatorManager(std::move(allocatorManager)) {
        glfwSetWindowUserPointer(windowInstance, this);
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

    std::vector<VertexType> vertices; /**< A vector holding vertex data. */ 
    std::vector<IndexType> indices; /**< A vector holding index data. */ 
    VkQueue graphicsQueue; /**< Vulkan queue for graphics operations. */ 
    VkQueue presentQueue; /**< Vulkan queue for presentation operations. */ 
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
    std::unique_ptr<AllocatorManager> allocatorManager; /**< Handles memory allocation for Vulkan objects. */ 
    
    uint32_t currentFrame = 0; /**< Tracks the current frame index for rendering synchronization. */ 
    bool framebufferResized = false; /**< Indicates if the framebuffer has been resized. */ 
// --------------------------------------------------------------------------------

    /**
     * @brief Cleans up Vulkan resources.
     * 
     * Resets all smart pointers and Vulkan handles to free memory and prevent resource leaks.
     * Called in the destructor.
     */
    void destroyResources() {
        glfwDestroyWindow(windowInstance);
        glfwTerminate();
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
