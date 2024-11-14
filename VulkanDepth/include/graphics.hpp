// ================================================================================
// ================================================================================
// - File:    graphics.hpp
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

#ifndef graphics_HPP
#define graphics_HPP

// Define Preprocessor Macros (before including related libraries)
#ifndef GLM_FORCE_RADIANS 
#define GLM_FORCE_RADIANS
#endif 

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE 
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif

// Standard Libraries
#include <vector>
#include <array>
#include <string>
#include <cstddef>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <filesystem>

// Vulkan and GLM Libraries
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>    // Vulkan Memory Allocator, often depends on Vulkan being included first
#include <glm/glm.hpp>        // GLM, typically safe after Vulkan
#include <glm/gtc/matrix_transform.hpp>

// stb_image.h Library
#include "stb_image.h"        
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif

// Project Headers
#include "memory.hpp"
#include "devices.hpp"
// ================================================================================
// ================================================================================ 

static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
// ================================================================================
// ================================================================================ 

/**
 * @brief Represents a vertex with position and color attributes.
 *
 * This struct defines a vertex with a 3D position and a 3D color. It also provides
 * static methods to describe how these vertex attributes are laid out in memory
 * for Vulkan's vertex input system. 
 */
struct ThreeDVertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
// -------------------------------------------------------------------------------- 

    /**
     * @brief Constructs a Vertex with the specified position, color, and texture coordinates.
     * 
     * @param pos Position of the vertex in 3D space.
     * @param color Color of the vertex as an RGB vector.
     * @param texCoord Texture coordinates of the vertex.
     */
    ThreeDVertex(const glm::vec3& pos, const glm::vec3& color, const glm::vec2& texCoord)
        : pos(pos), color(color), texCoord(texCoord) {}
// --------------------------------------------------------------------------------

    /**
     * @brief Returns the binding description for the vertex input.
     *
     * This function specifies how the vertex data is organized in the vertex buffer.
     * It provides the binding index, the byte stride between consecutive vertex data,
     * and the rate at which the input should advance.
     *
     * @return A VkVertexInputBindingDescription struct that describes the input binding for the vertex.
     */
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(ThreeDVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Returns the attribute descriptions for the vertex input.
     *
     * This function describes the layout of the vertex attributes (position, color, and texture coordinates) in memory.
     * It specifies the format of each attribute and the byte offset from the start of the vertex structure.
     *
     * @return A std::array of VkVertexInputAttributeDescription structs that describe the vertex attributes.
     */
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(ThreeDVertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(ThreeDVertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(ThreeDVertex, texCoord);

        return attributeDescriptions;
    }
};
// ================================================================================
// ================================================================================

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};
// ================================================================================
// ================================================================================

/**
 * @class DepthManager
 * @brief Manages the creation and handling of depth resources for Vulkan rendering.
 *
 * This class encapsulates the logic for creating, recreating, and managing Vulkan depth
 * buffers, including the associated image, memory allocation, and image view.
 */
class DepthManager {
public:
    
    /**
     * @brief Constructs a DepthManager instance.
     * 
     * Initializes the DepthManager with references to the AllocatorManager, Vulkan device,
     * physical device, and swap chain extent.
     *
     * @param allocatorManager A reference to the AllocatorManager for memory allocation.
     * @param device The Vulkan logical device.
     * @param physicalDevice The Vulkan physical device.
     * @param swapChainExtent The extent of the swap chain for creating depth resources.
     */
    DepthManager(AllocatorManager& allocatorManager, VkDevice device, 
                 VkPhysicalDevice physicalDevice, VkExtent2D swapChainExtent);
// --------------------------------------------------------------------------------

    /**
     * @brief Destructor for DepthManager.
     * 
     * Cleans up Vulkan depth image, memory, and image view resources.
     */
    ~DepthManager();
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the depth image view.
     * 
     * Provides access to the VkImageView used for depth buffering.
     *
     * @return The depth image view.
     */
    VkImageView getDepthImageView() const { return depthImageView; }
// --------------------------------------------------------------------------------

    /**
     * @brief Creates the depth resources.
     * 
     * Allocates and initializes the depth image, memory, and image view based on
     * the current swap chain extent and supported depth format.
     */
    void createDepthResources();
// --------------------------------------------------------------------------------
    
    /**
     * @brief Recreates the depth resources.
     * 
     * Cleans up existing depth resources and creates new ones with the given extent.
     *
     * @param newExtent The new swap chain extent to create depth resources for.
     */
    void recreateDepthResources(  VkExtent2D newExtent); 
// --------------------------------------------------------------------------------

    /**
     * @brief Finds the optimal depth format.
     * 
     * Determines a supported depth format from a list of preferred formats, checking
     * for depth-stencil attachment support.
     *
     * @return A supported VkFormat for depth buffering.
     */
    VkFormat findDepthFormat();
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the depth image view.
     *
     * @return The depth image view.
     */
    VkImageView getDepthImageView();
// ================================================================================
private:
    AllocatorManager& allocatorManager; ///< Reference to the AllocatorManager for memory allocation. 
    VkDevice device; ///< Vulkan logical device. 
    VkPhysicalDevice physicalDevice; ///< Vulkan physical device. 
    VkExtent2D swapChainExtent; ///< Current extent of the swap chain. 
    
    VkImage depthImage = VK_NULL_HANDLE; ///< Handle for the depth image. 
    VmaAllocation depthImageMemory = VK_NULL_HANDLE; ///< Handle for the memory allocation of the depth image. 
    VkImageView depthImageView = VK_NULL_HANDLE; ///< Handle for the depth image view. 
// --------------------------------------------------------------------------------

    /**
     * @brief Checks if a format has a stencil component.
     * 
     * Determines if a given format includes a stencil component.
     *
     * @param format The Vulkan format to check.
     * @return True if the format includes a stencil component, false otherwise.
     */
    bool hasStencilComponent(VkFormat format);
// -------------------------------------------------------------------------------- 

    /**
     * @brief Finds a supported format from a list of candidates.
     * 
     * Iterates through a list of candidate formats and returns the first one that meets
     * the specified tiling and feature requirements.
     *
     * @param candidates A list of VkFormat candidates.
     * @param tiling The tiling mode to check for (e.g., optimal or linear).
     * @param features The format feature flags to check for support.
     * @return The first supported VkFormat from the candidates.
     * @throws std::runtime_error if no supported format is found.
     */
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, 
                                 VkImageTiling tiling, VkFormatFeatureFlags features);
// --------------------------------------------------------------------------------

    /**
     * @brief Creates an image for depth buffering.
     * 
     * Initializes a Vulkan image for depth buffering, along with its memory allocation.
     *
     * @param width The width of the image.
     * @param height The height of the image.
     * @param format The format of the image.
     * @param tiling The tiling mode of the image.
     * @param usage The usage flags for the image.
     * @param memoryUsage The memory usage type (e.g., GPU-only).
     * @param image The image handle to be initialized.
     * @param imageMemory The memory allocation handle to be initialized.
     */
    void createImage(uint32_t width, uint32_t height, VkFormat format, 
                     VkImageTiling tiling, VkImageUsageFlags usage, 
                     VmaMemoryUsage memoryUsage, VkImage& image, 
                     VmaAllocation& imageMemory);
// --------------------------------------------------------------------------------

    /**
     * @brief Creates an image view for depth buffering.
     * 
     * Initializes a Vulkan image view for the provided image and format.
     *
     * @param image The Vulkan image to create the view for.
     * @param format The format of the image.
     * @param aspectFlags The aspect flags for the image view (e.g., depth).
     * @return The created VkImageView.
     * @throws std::runtime_error if image view creation fails.
     */
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
// --------------------------------------------------------------------------------

    /**
     * @brief Cleans up the depth resources.
     * 
     * Releases the depth image, memory, and image view resources.
     */
    void cleanup();
};
// ================================================================================
// ================================================================================

/**
 * @brief Manages Vulkan command buffers, synchronization objects, and command pool.
 * 
 * This class is responsible for creating and managing Vulkan command buffers, semaphores, 
 * and fences for synchronizing rendering operations. It also provides accessors to retrieve
 * synchronization objects for each frame in a multi-frame setup.
 * 
 * @tparam IndexType The type used for indexing, such as uint16_t or uint32_t.
 */
template <typename IndexType>
class CommandBufferManager {
public:

    /**
     * @brief Constructs the CommandBufferManager and initializes required resources.
     * 
     * Initializes command buffers, semaphores, fences, and the command pool needed for 
     * rendering operations. Resources are allocated based on the maximum frames in flight.
     * 
     * @param device The Vulkan device handle.
     * @param indices A reference to a vector containing index data for rendering.
     * @param physicalDevice The Vulkan physical device used to determine queue family indices.
     * @param surface The Vulkan surface associated with the rendering context.
     */
    CommandBufferManager(VkDevice device,
                         const std::vector<IndexType>& indices,
                         VkPhysicalDevice physicalDevice,
                         VkSurfaceKHR surface)
        : device(device),
          indices(indices) {
        // Create initial size for vectors
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT),
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT),
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        // Instantiate attributes
        createCommandPool(physicalDevice, surface);
        createSyncObjects();
        createCommandBuffers(); 
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Destructor for CommandBufferManager.
     * 
     * Cleans up Vulkan resources, including command buffers, semaphores, and fences.
     */
    ~CommandBufferManager() {
        if (device != VK_NULL_HANDLE) {
            // Free command buffers before destroying the command pool
            if (!commandBuffers.empty() && commandPool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
            }

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                if (imageAvailableSemaphores[i] != VK_NULL_HANDLE) {
                    vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
                }
                if (renderFinishedSemaphores[i] != VK_NULL_HANDLE) {
                    vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
                }
                if (inFlightFences[i] != VK_NULL_HANDLE) {
                    vkDestroyFence(device, inFlightFences[i], nullptr);
                }
            }

            if (commandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, commandPool, nullptr);
            }
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Waits for a specific frame’s fence to ensure frame completion.
     * 
     * Blocks until the fence for the given frame index is signaled, indicating 
     * that the frame rendering has finished.
     * 
     * @param frameIndex The index of the frame whose fence to wait on.
     * @throws std::runtime_error if waiting on the fence fails.
     */
    void waitForFences(uint32_t frameIndex) const {
        VkResult result = vkWaitForFences(device, 1, &inFlightFences[frameIndex], VK_TRUE, UINT64_MAX);
        std::string msg = std::string("Failed to wait for fence at frame index ") + 
                          std::to_string(frameIndex) + 
                          ". Error code: " + 
                          std::to_string(static_cast<int>(result));

        if (result != VK_SUCCESS) {
            throw std::runtime_error(msg);
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Resets a specific frame’s fence.
     * 
     * Resets the fence for the given frame index, allowing it to be used in the next frame.
     * 
     * @param frameIndex The index of the frame whose fence to reset.
     * @throws std::runtime_error if resetting the fence fails.
     */
    void resetFences(uint32_t frameIndex) const {
        VkResult result = vkResetFences(device, 1, &inFlightFences[frameIndex]);
        std::string msg = std::string("Failed to reset fence at frame index ") + 
                          std::to_string(frameIndex) + 
                          ". Error code: " + 
                          std::to_string(result);
        if (result != VK_SUCCESS) {
        throw std::runtime_error(msg); 
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the command pool used for allocating command buffers.
     * 
     * @return A reference to the VkCommandPool.
     * @throws std::runtime_error if the command pool is uninitialized.
     */
    const VkCommandPool& getCommandPool() const {
        if (commandPool == VK_NULL_HANDLE) {
            throw std::runtime_error("Command pool is not initialized.");
        }
        return commandPool;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Provides access to all allocated command buffers.
     * 
     * @return A reference to the vector of VkCommandBuffer.
     * @throws std::runtime_error if command buffers are not allocated.
     */
    const std::vector<VkCommandBuffer>& getCommandBuffers() const {
        if (commandBuffers.empty()) {
            throw std::runtime_error("Command buffers are not allocated.");
        }
        return commandBuffers;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves a specific command buffer for the given frame index.
     * 
     * @param frameIndex The index of the frame.
     * @return A reference to the VkCommandBuffer for the specified frame.
     * @throws std::runtime_error if the command buffer at the frame index is uninitialized.
     */
    const VkCommandBuffer& getCommandBuffer(uint32_t frameIndex) const {
        if (commandBuffers[frameIndex] == VK_NULL_HANDLE) {
            throw std::runtime_error(std::string("Command Buffer .") +
                                     std::to_string(frameIndex) +
                                     std::string(" does not exist!"));
        }
        return commandBuffers[frameIndex];
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the semaphore signaling image availability for a specific frame.
     * 
     * @param frameIndex The index of the frame.
     * @return A reference to the VkSemaphore signaling image availability.
     * @throws std::runtime_error if the semaphore at the frame index is uninitialized.
     */
    const VkSemaphore& getImageAvailableSemaphore(uint32_t frameIndex) const {
        if (imageAvailableSemaphores[frameIndex] == VK_NULL_HANDLE)
            throw std::runtime_error(std::string("Image available semaphore ") +
                                                 std::to_string(frameIndex) +
                                                 std::string(" does not exist!"));
        return imageAvailableSemaphores[frameIndex];
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the semaphore signaling render completion for a specific frame.
     * 
     * @param frameIndex The index of the frame.
     * @return A reference to the VkSemaphore signaling render completion.
     * @throws std::runtime_error if the semaphore at the frame index is uninitialized.
     */
    const VkSemaphore& getRenderFinishedSemaphore(uint32_t frameIndex) const {
        if (renderFinishedSemaphores[frameIndex] == VK_NULL_HANDLE)
            throw std::runtime_error(std::string("Render Finished Semaphore ") +
                                     std::to_string(frameIndex) +
                                     std::string(" does not exist!"));
        return renderFinishedSemaphores[frameIndex];
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the in-flight fence for a specific frame.
     * 
     * The fence is used to track the rendering state of each frame.
     * 
     * @param frameIndex The index of the frame.
     * @return A reference to the VkFence for the specified frame.
     * @throws std::runtime_error if the fence at the frame index is uninitialized.
     */
    const VkFence& getInFlightFence(uint32_t frameIndex) const {
        if (inFlightFences[frameIndex] == VK_NULL_HANDLE)
            throw std::runtime_error(std::string("In Flight Fence ") +
                                     std::to_string(frameIndex) +
                                     std::string(" does not exist!"));
        return inFlightFences[frameIndex];
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Allocates command buffers for rendering operations.
     * 
     * Command buffers are allocated from the command pool and used for recording rendering commands.
     * 
     * @throws std::runtime_error if command buffer allocation fails.
     */
    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        VkResult result = vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
        std::string msg = std::string("Failed to create command pool!: Error code: ") + 
                          std::to_string((result));
        if (result != VK_SUCCESS) {
            throw std::runtime_error(msg);
        }  
    }
// ================================================================================
private:
    // Attributes passed to constructor
    VkDevice device;                      /**< The Vulkan device handle. */
    VkExtent2D swapChainExtent;           /**< The extent of the swap chain for rendering. */
    std::vector<IndexType> indices;        /**< Vector holding index data for command buffers. */

    VkCommandPool commandPool = VK_NULL_HANDLE; /**< The Vulkan command pool used to allocate command buffers. */
    std::vector<VkCommandBuffer> commandBuffers; /**< The list of Vulkan command buffers. */
    std::vector<VkSemaphore> imageAvailableSemaphores; /**< Semaphores used to signal when images are available. */
    std::vector<VkSemaphore> renderFinishedSemaphores; /**< Semaphores used to signal when rendering is finished. */
    std::vector<VkFence> inFlightFences; /**< Fences used for synchronizing frame rendering. */ 
// --------------------------------------------------------------------------------

    /**
     * @brief Creates a Vulkan command pool for command buffer allocation.
     * 
     * The command pool is associated with the graphics queue family.
     * 
     * @param physicalDevice The Vulkan physical device.
     * @param surface The surface to query queue family support.
     * @throws std::runtime_error if command pool creation fails.
     */
    void createCommandPool(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
        QueueFamilyIndices queueFamilyIndices = QueueFamily::findQueueFamilies(physicalDevice, surface);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
        std::string msg = std::string("Failed to create command pool!: Error code: ") + 
                          std::to_string((result));
        if (result != VK_SUCCESS) {
            throw std::runtime_error(msg);
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Creates semaphores and fences for synchronizing rendering.
     * 
     * Allocates semaphores for image availability and render completion, and fences for tracking in-flight frames.
     * @throws std::runtime_error if semaphore or fence creation fails.
     */
    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        auto createSemaphore = [this]() -> VkSemaphore {
            VkSemaphore semaphore;
            VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
                throw std::runtime_error("failed to create semaphore!");
            }
            return semaphore;
        };

        auto createFence = [this]() -> VkFence {
            VkFence fence;
            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
                throw std::runtime_error("failed to create fence!");
            }
            return fence;
        };

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            imageAvailableSemaphores[i] = createSemaphore();
            renderFinishedSemaphores[i] = createSemaphore();
            inFlightFences[i] = createFence();
        }
    }
};
// ================================================================================
// ================================================================================ 

/**
 * @class SamplerManager
 * @brief Manages the creation, retrieval, and destruction of Vulkan texture samplers.
 *
 * The SamplerManager class is responsible for managing Vulkan samplers. 
 * It allows for the creation of reusable samplers based on unique keys, 
 * making it efficient to share samplers across multiple textures that have similar sampling properties.
 */
class SamplerManager {
public:

    /**
     * @brief Constructs a SamplerManager for managing Vulkan texture samplers.
     *
     * Initializes the SamplerManager with the Vulkan device and physical device handles needed
     * for creating samplers.
     * 
     * @param device The Vulkan logical device handle used to create and manage samplers.
     * @param physicalDevice The Vulkan physical device handle used for querying properties 
     *        such as supported anisotropy levels.
     */
    SamplerManager(VkDevice device, VkPhysicalDevice physicalDevice);
// --------------------------------------------------------------------------------

    /**
     * @brief Destructor for the SamplerManager class.
     *
     * Cleans up all Vulkan samplers created and managed by the SamplerManager.
     * Each sampler in the manager is destroyed upon the SamplerManager's destruction.
     */
    ~SamplerManager();
// --------------------------------------------------------------------------------
    /**
     * @brief Retrieves an existing sampler by key.
     *
     * Looks up the sampler associated with the specified key. If the sampler does not exist,
     * it will throw an exception.
     * 
     * @param samplerKey A unique key identifying the desired sampler.
     * @return The Vulkan sampler associated with the given key.
     * @throws std::out_of_range if the sampler associated with the key does not exist.
     */
    VkSampler getSampler(const std::string& samplerKey) const;
// --------------------------------------------------------------------------------
    /**
     * @brief Creates and stores a sampler with the specified key.
     *
     * Creates a new Vulkan sampler with commonly used sampling parameters and associates it
     * with the specified key for future retrieval. The created sampler is stored in the 
     * `samplers` map and can be reused across multiple textures.
     * 
     * @param samplerKey A unique key identifying the sampler for future retrieval.
     * @return The Vulkan sampler created and stored with the specified key.
     * @throws std::runtime_error if Vulkan fails to create the sampler.
     */
    VkSampler createSampler(const std::string& samplerKey);
// ================================================================================
private:

    VkDevice device; /**< The Vulkan logical device handle used for sampler creation. */ 
    VkPhysicalDevice physicalDevice; /**< The Vulkan physical device handle for querying properties. */ 
    std::unordered_map<std::string, VkSampler> samplers;  /**< Map of samplers keyed by unique strings for reuse. */

    mutable std::mutex samplerMutex; /**< Mutex to synchronize access to the `samplers` map. */ 
};
// ================================================================================
// ================================================================================ 

template <typename IndexType>
class TextureManager {
public:
    /**
     * @brief Constructs a TextureManager instance to manage texture creation, memory allocation, and transitions.
     * 
     * Initializes and loads a texture image from the specified file path, sets up an image view, and retrieves 
     * a texture sampler from the SamplerManager.
     *
     * @param allocatorManager Reference to an AllocatorManager instance for handling Vulkan memory allocations.
     * @param device The Vulkan logical device used for creating and managing resources.
     * @param physicalDevice The Vulkan physical device, used for querying memory properties.
     * @param commandBufferManager Reference to a CommandBufferManager to handle command buffer allocations.
     * @param graphicsQueue The graphics queue used to submit commands for texture transfer operations.
     * @param image The file path of the texture image to be loaded.
     * @param samplerManager Reference to the SamplerManager to retrieve a texture sampler.
     * @param samplerKey The key identifier for the desired texture sampler in the SamplerManager.
     */
   TextureManager(AllocatorManager& allocatorManager,
                  VkDevice device,
                  VkPhysicalDevice physicalDevice,
                  CommandBufferManager<IndexType>& commandBufferManager,
                  VkQueue graphicsQueue,
                  const std::string image,
                  SamplerManager& samplerManager,
                  const std::string& samplerKey)
        : allocatorManager(allocatorManager),
          device(device),
          physicalDevice(physicalDevice),
          commandBufferManager(commandBufferManager),
          graphicsQueue(graphicsQueue),
          imagePath(image){
        createTextureImage();
        createTextureImageView();
        textureSampler = samplerManager.getSampler(samplerKey);
    } 
// --------------------------------------------------------------------------------

   /**
     * @brief Destructor for TextureManager.
     * 
     * Cleans up the texture image view and Vulkan image resources allocated by this manager.
     */
    ~TextureManager() {
    
        if (textureImageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, textureImageView, nullptr);
            textureImageView = VK_NULL_HANDLE;
        }

        if (textureImage != VK_NULL_HANDLE && textureImageMemory != VK_NULL_HANDLE) {
            vmaDestroyImage(allocatorManager.getAllocator(), textureImage, textureImageMemory);
            textureImage = VK_NULL_HANDLE;
            textureImageMemory = VK_NULL_HANDLE;
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Returns the Vulkan image view associated with the texture.
     * 
     * @return The VkImageView of the texture, to be used for rendering.
     */

    VkImageView getTextureImageView() const { return textureImageView; }
// --------------------------------------------------------------------------------

    /**
     * @brief Reloads the texture from a new image file.
     * 
     * Replaces the existing texture with a new image, releasing previous resources and reinitializing the texture.
     *
     * @param newImagePath The file path to the new image file for the texture.
     */
    void reloadTexture(const std::string& newImagePath) {
        std::lock_guard<std::mutex> lock(textureMutex);  // Ensures no other thread can modify the texture during reload

        // Safely release existing resources
        if (textureImage != VK_NULL_HANDLE && textureImageMemory != VK_NULL_HANDLE) {
            vmaDestroyImage(allocatorManager.getAllocator(), textureImage, textureImageMemory);
            textureImage = VK_NULL_HANDLE;
            textureImageMemory = VK_NULL_HANDLE;
        }

        if (textureImageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, textureImageView, nullptr);
            textureImageView = VK_NULL_HANDLE;
        }

        // Update image path and recreate texture
        imagePath = newImagePath;
        createTextureImage();
        createTextureImageView();
    }
// ===============================================================================
private:
    AllocatorManager& allocatorManager;  /**< The memory allocator manager for handling buffer memory. */
    VkDevice device;                     /**< The Vulkan device handle. */
    VkPhysicalDevice physicalDevice; /**< The Vulkan physical device used for querying memory properties. */ 
    CommandBufferManager<IndexType>& commandBufferManager;  /**< Reference to the command buffer manager. */
    VkQueue graphicsQueue;/**< The Vulkan graphics queue used for submitting commands. */ 
    std::string imagePath; /**< Path to the texture image to be loaded. */ 

    VkImage textureImage = VK_NULL_HANDLE ; /**< The Vulkan image object representing the texture. */ 
    VmaAllocation textureImageMemory = VK_NULL_HANDLE ; /**< The memory backing the Vulkan texture image. */ 
    VkImageView textureImageView = VK_NULL_HANDLE;
    VkSampler textureSampler = VK_NULL_HANDLE;

    std::mutex textureMutex;
// --------------------------------------------------------------------------------

    /**
     * @brief Creates the texture image from the loaded image file.
     * 
     * Allocates memory and uploads pixel data from the CPU to the GPU, then sets up the image layout
     * for shader sampling.
     * 
     * @throws std::runtime_error If the texture image cannot be loaded or memory mapping fails.
     */
    void createTextureImage() {
        std::lock_guard<std::mutex> lock(textureMutex);
        if (textureImage != VK_NULL_HANDLE && textureImageMemory != VK_NULL_HANDLE) {
            return; // Resources already initialized, skip re-creation
        }

        if (imagePath.empty()) {
            throw std::invalid_argument("TextureManager: imagePath is empty, please provide a valid texture file path.");
        }
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(imagePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("Failed to load texture image!");
        }

        // Create a staging buffer with VMA
        VkBuffer stagingBuffer;
        VmaAllocation stagingBufferMemory;
        allocatorManager.createBuffer(
            imageSize, 
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VMA_MEMORY_USAGE_CPU_TO_GPU, // Use VMA-specific memory usage for CPU to GPU transfer
            stagingBuffer, 
            stagingBufferMemory
        );

        // Map memory, copy pixel data to the staging buffer, then unmap memory
        void* data;
        if (vmaMapMemory(allocatorManager.getAllocator(), stagingBufferMemory, &data) != VK_SUCCESS) {
            throw std::runtime_error("TextureManager::createTextureImage: Failed to map memory for staging buffer.");
        }
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vmaUnmapMemory(allocatorManager.getAllocator(), stagingBufferMemory);

        stbi_image_free(pixels); // Free the loaded image pixels now that they're in the staging buffer

        // Create the texture image on the GPU
        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, 
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
                    VMA_MEMORY_USAGE_GPU_ONLY, textureImage, textureImageMemory);
    
        // Transition the texture image layout and copy data from the staging buffer
        transitionImageLayout(
            textureImage, 
            VK_FORMAT_R8G8B8A8_SRGB, 
            VK_IMAGE_LAYOUT_UNDEFINED, 
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        );
        copyBufferToImage(
            stagingBuffer, 
            textureImage, 
            static_cast<uint32_t>(texWidth), 
            static_cast<uint32_t>(texHeight)
        );
        transitionImageLayout(
            textureImage, 
            VK_FORMAT_R8G8B8A8_SRGB, 
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );

        // Clean up the staging buffer using VMA
        vmaDestroyBuffer(allocatorManager.getAllocator(), stagingBuffer, stagingBufferMemory);
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Helper function to create a Vulkan image with the specified properties.
     * 
     * Allocates and initializes a Vulkan image with the desired format, tiling, and usage.
     *
     * @param width Width of the image in pixels.
     * @param height Height of the image in pixels.
     * @param format The Vulkan format for the image.
     * @param tiling Tiling arrangement (optimal or linear).
     * @param usage The usage flags defining how the image is accessed.
     * @param memoryUsage Memory usage pattern (e.g., GPU-only).
     * @param image Output Vulkan image handle.
     * @param imageMemory Output allocation handle for the image.
     * @throws std::runtime_error If image creation fails.
     */
    void createImage(uint32_t width, uint32_t height, VkFormat format, 
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
            throw std::runtime_error(
                std::string("TextureManager::createImage: Failed to create image with properties:\n") +
                " Width: " + std::to_string(width) + 
                ", Height: " + std::to_string(height) + 
                ", Format: " + std::to_string(format) + 
                ", Usage: " + std::to_string(usage)
            );
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Finds a suitable memory type for the texture image.
     * 
     * Queries memory properties and selects a memory type with the desired properties.
     * 
     * @param typeFilter Bitmask specifying memory type requirements.
     * @param properties Required memory property flags.
     * @return Index of the suitable memory type.
     * @throws std::runtime_error If no suitable memory type is found.
     */
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    } 
// --------------------------------------------------------------------------------

    /**
     * @brief Begins a single-time command buffer for executing short-lived commands.
     * 
     * Allocates and starts recording a command buffer intended for immediate execution.
     * 
     * @return The allocated command buffer.
     */
    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBuffer commandBuffer = allocateCommandBuffer();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        return commandBuffer;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Ends and submits a single-time command buffer.
     * 
     * Finishes recording the command buffer, submits it, and waits for completion.
     * 
     * @param commandBuffer The command buffer to be ended and submitted.
     */
    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);
        submitSingleTimeCommandBuffer(commandBuffer);
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Transitions the layout of a Vulkan image.
     * 
     * Configures the layout of a Vulkan image between various stages (e.g., from transfer to shader-read).
     * 
     * @param image The Vulkan image to transition.
     * @param format The format of the image.
     * @param oldLayout The current layout of the image.
     * @param newLayout The desired new layout for the image.
     * @throws std::invalid_argument If the layout transition is unsupported.
     */
    void transitionImageLayout(
        VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier = createImageMemoryBarrier(image, format, oldLayout, newLayout);

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands(commandBuffer);
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Copies data from a buffer to an image on the GPU.
     * 
     * Transfers pixel data from a staging buffer to the texture image.
     * 
     * @param buffer The Vulkan buffer containing the pixel data.
     * @param image The Vulkan image receiving the data.
     * @param width The width of the image in pixels.
     * @param height The height of the image in pixels.
     */
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
            VkCommandBuffer commandBuffer = beginSingleTimeCommands();

            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {
                width,
                height,
                1
            };

            vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            endSingleTimeCommands(commandBuffer);
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Determines if a Vulkan format includes a stencil component.
     * 
     * @param format The Vulkan format to check.
     * @return True if the format has a stencil component, false otherwise.
     */
    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Creates an image view for accessing the texture.
     * 
     * Sets up a Vulkan image view to access the texture for sampling.
     */
    void createTextureImageView() {
        if (textureImageView != VK_NULL_HANDLE) {
            return; // Image view already exists, skip re-creation
        }
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Creates a Vulkan image view with specified properties.
     * 
     * @param image The Vulkan image to view.
     * @param format The format of the image view.
     * @return The created Vulkan image view.
     * @throws std::runtime_error If image view creation fails.
     */
    VkImageView createImageView(VkImage image, VkFormat format) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
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

    /**
     * @brief Creates an image memory barrier for transitioning image layouts.
     * 
     * Configures an image memory barrier with specified properties for use in layout transitions.
     * 
     * @param image The Vulkan image for the barrier.
     * @param format The format of the image.
     * @param oldLayout The current layout of the image.
     * @param newLayout The target layout of the image.
     * @return A configured VkImageMemoryBarrier for the transition.
     */
    VkImageMemoryBarrier createImageMemoryBarrier(
        VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;

        if (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (hasStencilComponent(format)) {
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        } else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }

        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        return barrier;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Allocates a primary command buffer for a one-time command.
     * 
     * @return The allocated command buffer.
     */
    VkCommandBuffer allocateCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandBufferManager.getCommandPool();
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
        return commandBuffer;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Submits and waits for a single-time command buffer.
     * 
     * Submits the command buffer to the graphics queue and waits until execution completes, then releases it.
     * 
     * @param commandBuffer The command buffer to submit.
     */
    void submitSingleTimeCommandBuffer(VkCommandBuffer commandBuffer) {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandBufferManager.getCommandPool(), 1, &commandBuffer);
    }
};
// ================================================================================
// ================================================================================ 

/**
 * @brief A template class that manages Vulkan buffers for vertex, index, and uniform data.
 * 
 * This class encapsulates the creation, management, and cleanup of Vulkan buffers
 * required for rendering, including vertex and index buffers for geometry, and uniform
 * buffers for per-frame data updates.
 * 
 * @tparam VertexType The type of vertex data used in the application.
 * @tparam IndexType The type of index data, supporting uint8_t, uint16_t, or uint32_t.
 */
template <typename VertexType, typename IndexType>
class BufferManager {
public:

    /**
     * @brief Constructs a BufferManager instance.
     * 
     * Initializes the BufferManager with vertices and indices, allocates buffers for
     * vertex, index, and uniform data, and configures them for rendering.
     * 
     * @param vertices The vertex data used for rendering.
     * @param indices The index data for drawing elements.
     * @param commandBufferManager A reference to the CommandBufferManager for buffer commands.
     * @param graphicsQueue The Vulkan queue used for submitting graphics commands.
     */
    BufferManager(const std::vector<VertexType>& vertices,
                  const std::vector<IndexType>& indices,
                  AllocatorManager& allocatorManager,
                  CommandBufferManager<IndexType>& commandBufferManager,
                  VkQueue graphicsQueue)
    : vertices(vertices),
      indices(indices),
      allocatorManager(allocatorManager),
      commandBufferManager(commandBufferManager),
      graphicsQueue(graphicsQueue) {
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Destructor for BufferManager.
     * 
     * Cleans up all allocated buffers, unmaps memory if mapped, and releases buffer memory.
     */
    ~BufferManager() {
        // Clean up uniform buffers
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (uniformBuffers[i] != VK_NULL_HANDLE) {
                // Unmap the memory if it was mapped
                if (uniformBuffersMapped[i] != nullptr) {
                    vmaUnmapMemory(allocatorManager.getAllocator(), uniformBuffersMemory[i]);
                }
                // Destroy the buffer and free the associated memory
                allocatorManager.destroyBuffer(uniformBuffers[i], uniformBuffersMemory[i]);
            }
        }

        // Clean up vertex buffer
        if (vertexBuffer != VK_NULL_HANDLE) {
            allocatorManager.destroyBuffer(vertexBuffer, vertexBufferAllocation);
        }

        // Clean up index buffer
        if (indexBuffer != VK_NULL_HANDLE) {
            allocatorManager.destroyBuffer(indexBuffer, indexBufferAllocation);
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Updates the uniform buffer for a given frame.
     * 
     * Maps the buffer memory for the specified frame, allowing the application to update
     * uniform data with new transformation matrices or other frame-specific data.
     * 
     * @param currentFrame The index of the frame to update.
     * @param ubo The UniformBufferObject containing transformation data for the current frame.
     * @throws std::out_of_range if currentFrame is beyond the bounds of MAX_FRAMES_IN_FLIGHT.
     * @throws std::runtime_error if the buffer memory cannot be mapped.
     */
    void updateUniformBuffer(uint32_t currentFrame, const UniformBufferObject& ubo) {
        // Ensure that the current frame index is within bounds
        if (currentFrame >= MAX_FRAMES_IN_FLIGHT) {
            throw std::out_of_range("Frame index out of bounds.");
        }

        // Map the memory for the current frame's uniform buffer
        void* data = uniformBuffersMapped[currentFrame];
        if (data == nullptr) {
            throw std::runtime_error(std::string("Failed to map uniform buffer for frame ") + std::to_string(currentFrame));
        }

        // Copy the UniformBufferObject data into the mapped memory
        memcpy(data, &ubo, sizeof(ubo));
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the Vulkan vertex buffer.
     * 
     * Provides access to the vertex buffer used for rendering.
     * 
     * @return The Vulkan buffer containing vertex data.
     */
    const VkBuffer getVertexBuffer() const{
        return vertexBuffer;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the Vulkan index buffer.
     * 
     * Provides access to the index buffer used for indexed drawing.
     * 
     * @return The Vulkan buffer containing index data.
     */
    const VkBuffer getIndexBuffer() const {
        return indexBuffer;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the uniform buffers for all frames.
     * 
     * Provides access to a vector of Vulkan uniform buffers for each frame.
     * 
     * @return A vector of Vulkan buffers used for uniform data.
     */
    const std::vector<VkBuffer>& getUniformBuffers() const {
        return uniformBuffers;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the mapped memory for the uniform buffers.
     * 
     * Provides access to a vector of pointers mapped to each frame's uniform buffer, allowing
     * for direct memory access to update uniform data.
     * 
     * @return A vector of pointers to the mapped uniform buffers.
     */
    const std::vector<void*>& getUniformBuffersMapped() const {
        return uniformBuffersMapped;
    }
// ================================================================================
private:
    std::vector<VertexType> vertices;  /**< The vertex data used for rendering. */
    std::vector<IndexType> indices;  /**< The index data for drawing elements. */
    AllocatorManager& allocatorManager;  /**< The memory allocator manager for handling buffer memory. */
    CommandBufferManager<IndexType>& commandBufferManager;  /**< Command buffer manager for managing related command buffers. */
    VkQueue graphicsQueue;  /**< The Vulkan queue used for submitting graphics commands. */

    VkBuffer vertexBuffer; /**< Vulkan buffer for storing vertex data. */
    VkBuffer indexBuffer;  /**< Vulkan buffer for storing index data. */
    VmaAllocation vertexBufferAllocation;  /**< Memory allocation handle for the vertex buffer. */
    VmaAllocation indexBufferAllocation;  /**< Memory allocation handle for the index buffer. */

    std::vector<VkBuffer> uniformBuffers;  /**< Vector of Vulkan buffers used for uniform data across frames. */
    std::vector<void*> uniformBuffersMapped;  /**< Vector of pointers that map uniform buffers for direct memory access. */
    std::vector<VmaAllocation> uniformBuffersMemory;  /**< Memory allocation handles for the uniform buffers. */
// --------------------------------------------------------------------------------

    /**
     * @brief Creates the vertex buffer.
     * 
     * Allocates and initializes a vertex buffer for GPU access, using a staging buffer to
     * transfer data from CPU to GPU memory.
     * 
     * @return True if the vertex buffer is successfully created, false otherwise.
     */
    bool createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        // Step 1: Create a staging buffer
        VkBuffer stagingBuffer;
        VmaAllocation stagingBufferAllocation;
        try {
            allocatorManager.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                                          VMA_MEMORY_USAGE_CPU_ONLY, stagingBuffer, stagingBufferAllocation);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return false;
        }

        // Step 2: Map memory and copy vertex data to the staging buffer
        void* data;
        try {
            allocatorManager.mapMemory(stagingBufferAllocation, &data);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            allocatorManager.destroyBuffer(stagingBuffer, stagingBufferAllocation); // Cleanup
            return false;
        }
        memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
        allocatorManager.unmapMemory(stagingBufferAllocation);

        // Step 3: Create the vertex buffer on the GPU
        try {
            allocatorManager.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
                                          VMA_MEMORY_USAGE_GPU_ONLY, vertexBuffer, vertexBufferAllocation);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            allocatorManager.destroyBuffer(stagingBuffer, stagingBufferAllocation); // Cleanup
            return false;
        }

        // Step 4: Copy data from the staging buffer to the vertex buffer
        try {
            allocatorManager.copyBuffer(stagingBuffer, vertexBuffer, bufferSize, graphicsQueue, commandBufferManager.getCommandPool());
        } catch (const std::runtime_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            allocatorManager.destroyBuffer(stagingBuffer, stagingBufferAllocation); // Cleanup
            allocatorManager.destroyBuffer(vertexBuffer, vertexBufferAllocation);   // Cleanup
            return false;
        }

        // Step 5: Clean up the staging buffer
        allocatorManager.destroyBuffer(stagingBuffer, stagingBufferAllocation);

        return true; // Indicate success
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Creates the index buffer.
     * 
     * Allocates and initializes an index buffer for GPU access, using a staging buffer to
     * transfer data from CPU to GPU memory.
     * 
     * @return True if the index buffer is successfully created, false otherwise.
     */
    bool createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        // Step 1: Create a staging buffer
        VkBuffer stagingBuffer;
        VmaAllocation stagingBufferAllocation;
        try {
            allocatorManager.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                                          VMA_MEMORY_USAGE_CPU_ONLY, stagingBuffer, stagingBufferAllocation);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return false;
        }

        // Step 2: Map memory and copy index data to the staging buffer
        void* data;
        try {
            allocatorManager.mapMemory(stagingBufferAllocation, &data);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            allocatorManager.destroyBuffer(stagingBuffer, stagingBufferAllocation); // Cleanup
            return false;
        }
        memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
        allocatorManager.unmapMemory(stagingBufferAllocation);

        // Step 3: Create the index buffer on the GPU
        try {
            allocatorManager.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
                                          VMA_MEMORY_USAGE_GPU_ONLY, indexBuffer, indexBufferAllocation);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            allocatorManager.destroyBuffer(stagingBuffer, stagingBufferAllocation); // Cleanup
            return false;
        }

        // Step 4: Copy data from the staging buffer to the index buffer
        try {
            allocatorManager.copyBuffer(stagingBuffer, indexBuffer, bufferSize, graphicsQueue, commandBufferManager.getCommandPool());
        } catch (const std::runtime_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            allocatorManager.destroyBuffer(stagingBuffer, stagingBufferAllocation); // Cleanup
            allocatorManager.destroyBuffer(indexBuffer, indexBufferAllocation);     // Cleanup
            return false;
        }

        // Step 5: Clean up the staging buffer
        allocatorManager.destroyBuffer(stagingBuffer, stagingBufferAllocation);

        return true; // Indicate success
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Creates uniform buffers for each frame in flight.
     * 
     * Allocates buffers for each frame to store per-frame uniform data and maps them to CPU
     * accessible memory for efficient updates.
     * 
     * @return True if the uniform buffers are successfully created, false otherwise.
     */
    bool createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        // Resize the buffers to the correct size for the number of frames in flight
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            try {
                // Step 1: Create a uniform buffer for each frame
                allocatorManager.createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                                              VMA_MEMORY_USAGE_CPU_TO_GPU, 
                                              uniformBuffers[i], uniformBuffersMemory[i]);
            } catch (const std::runtime_error& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                // Cleanup for previously created buffers
                for (size_t j = 0; j < i; j++) {
                    allocatorManager.destroyBuffer(uniformBuffers[j], uniformBuffersMemory[j]);
                }
                return false;
            }

            try {
                // Step 2: Map the uniform buffer memory
                allocatorManager.mapMemory(uniformBuffersMemory[i], &uniformBuffersMapped[i]);
            } catch (const std::runtime_error& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                // Cleanup for previously created buffers
                for (size_t j = 0; j <= i; j++) {
                    allocatorManager.destroyBuffer(uniformBuffers[j], uniformBuffersMemory[j]);
                }
                return false;
            }
        }

        return true; // Indicate success
    }
};
// ================================================================================
// ================================================================================ 

/**
 * @class DescriptorManager
 * @brief Manages Vulkan descriptor sets, layouts, and descriptor pools.
 *
 * This class handles the creation and management of Vulkan descriptor sets, layouts, and descriptor pools.
 * It is responsible for creating the descriptor sets for each frame and managing the descriptor pool and layout.
 */
class DescriptorManager {
public:
    /**
     * @brief Constructor for DescriptorManager.
     *
     * @param device The Vulkan device handle used for creating descriptor sets and pools.
     */
    DescriptorManager(VkDevice device);
// --------------------------------------------------------------------------------

    /**
     * @brief Destructor for DescriptorManager.
     *
     * Cleans up the descriptor pool and descriptor set layout associated with the manager.
     */
    ~DescriptorManager();
// --------------------------------------------------------------------------------

    /**
     * @brief Creates descriptor sets for each frame in the application.
     *
     * This method allocates and configures descriptor sets for each frame, allowing the shaders
     * to access uniform buffer data and texture sampling resources. The descriptor sets
     * are configured to include both the uniform buffer and the texture sampler.
     * 
     * @param uniformBuffers A vector of Vulkan buffers that hold the uniform buffer data for each frame.
     * @param textureImageView The Vulkan image view of the texture to be sampled in the shader.
     * @param textureSampler The Vulkan sampler used to sample the texture image.
     * 
     * @throws std::runtime_error if the descriptor sets cannot be allocated or updated.
     */ 
    void createDescriptorSets(const std::vector<VkBuffer> uniformBuffers,
                              VkImageView textureImageView,
                              VkSampler textureSampler);
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the Vulkan descriptor set layout.
     *
     * @return A reference to the Vulkan descriptor set layout.
     */
    const VkDescriptorSetLayout& getDescriptorSetLayout() const;
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the Vulkan descriptor pool.
     *
     * @return A reference to the Vulkan descriptor pool.
     */
    const VkDescriptorPool& getDescriptorPool() const;
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the vector of Vulkan descriptor sets.
     *
     * @return A reference to the vector of Vulkan descriptor sets.
     */
    const std::vector<VkDescriptorSet>& getDescriptorSets() const;
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the descriptor set for a specific frame.
     *
     * @param frameIndex The index of the frame for which to retrieve the descriptor set.
     * @return A reference to the Vulkan descriptor set for the given frame.
     */
    const VkDescriptorSet& getDescriptorSet(uint32_t frameIndex) const;
// ================================================================================
private:
    VkDevice device;                                /**< The Vulkan device handle. */

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;  /**< The layout of the descriptor sets. */
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;            /**< The descriptor pool for allocating descriptor sets. */
    std::vector<VkDescriptorSet> descriptorSets;                 /**< A vector holding descriptor sets for each frame. */
// --------------------------------------------------------------------------------

    /**
     * @brief Creates the Vulkan descriptor set layout.
     *
     * Defines the layout of the descriptor sets and creates a corresponding layout object.
     */
    void createDescriptorSetLayout();
// --------------------------------------------------------------------------------

    /**
     * @brief Creates the Vulkan descriptor pool.
     *
     * Allocates a descriptor pool from which descriptor sets can be allocated.
     */
    void createDescriptorPool();
};
// ================================================================================
// ================================================================================ 

/**
 * @brief Manages Vulkan graphics pipeline creation and related resources.
 * 
 * This class handles the setup and configuration of Vulkan graphics pipeline stages,
 * including shaders, render passes, framebuffers, and command buffer recording.
 * The pipeline is designed to support custom vertex and index types for flexibility.
 * 
 * @tparam VertexType The type of vertex data used in the application.
 * @tparam IndexType The type of index data used, typically uint16_t, uint32_t, or uint8_t.
 */
template <typename VertexType, typename IndexType>
class GraphicsPipeline {
public:

    /**
     * @brief Constructs a GraphicsPipeline instance.
     * 
     * Initializes the graphics pipeline with the specified parameters and creates
     * the render pass and pipeline stages.
     * 
     * @param device The Vulkan logical device.
     * @param swapChain Reference to the swap chain used for rendering.
     * @param commandBufferManager Reference to the CommandBufferManager.
     * @param bufferManager Reference to the BufferManager holding vertex and index buffers.
     * @param descriptorManager Reference to the DescriptorManager.
     * @param indices Vector of indices for drawing.
     * @param physicalDevice The Vulkan physical device.
     * @param vertFile Path to the vertex shader file.
     * @param fragFile Path to the fragment shader file.
     * @param depthManager Reference to the DepthManager for depth resources.
     */
    GraphicsPipeline(VkDevice device,
                     SwapChain& swapChain,
                     CommandBufferManager<IndexType>& commandBufferManager,
                     BufferManager<VertexType, IndexType>& bufferManager,
                     DescriptorManager& descriptorManager,
                     const std::vector<IndexType>& indices,
                     VulkanPhysicalDevice& physicalDevice,
                     std::string vertFile,
                     std::string fragFile,
                     DepthManager& depthManager)
        : device(device),
          swapChain(swapChain),
          commandBufferManager(commandBufferManager),
          bufferManager(bufferManager),
          descriptorManager(descriptorManager),
          indices(indices),
          physicalDevice(physicalDevice),
          vertFile(vertFile),
          fragFile(fragFile),
          depthManager(depthManager) {
       createRenderPass(swapChain.getSwapChainImageFormat());
       createGraphicsPipeline();
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Destructor for GraphicsPipeline.
     * 
     * Cleans up Vulkan pipeline resources, including framebuffers, pipeline layout,
     * graphics pipeline, and render pass.
     */
    ~GraphicsPipeline() {
        // Destroy framebuffers explicitly
        destroyFramebuffers();

        // Clean up the graphics pipeline, pipeline layout, and render pass
        if (graphicsPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, graphicsPipeline, nullptr);
        }
        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        }
        if (renderPass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(device, renderPass, nullptr);
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Creates framebuffers for the swap chain images and depth attachment.
     * 
     * Configures framebuffers for each swap chain image, attaching color and depth views.
     * 
     * @param swapChainImageViews Vector of image views for each swap chain image.
     * @param swapChainExtent Extent of the swap chain, specifying width and height.
     * @throws std::runtime_error if the depth image view is not initialized.
     */
    void createFrameBuffers(const std::vector<VkImageView>& swapChainImageViews, 
                       VkExtent2D swapChainExtent) {
        // Verify depthImageView is initialized
        VkImageView depthImageView = depthManager.getDepthImageView();
        if (depthImageView == VK_NULL_HANDLE) {
            throw std::runtime_error("DepthManager depthImageView is not initialized.");
        }
        framebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                swapChainImageViews[i],
                depthImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Destroys all framebuffers created by the pipeline.
     */
    void destroyFramebuffers() {
        for (auto framebuffer : framebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        framebuffers.clear();
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Records commands to render a frame into a command buffer.
     * 
     * Sets up rendering state, binds pipeline, vertex and index buffers, descriptor sets,
     * and initiates indexed drawing of primitives.
     * 
     * @param frameIndex Index of the current frame in flight.
     * @param imageIndex Index of the swap chain image.
     * @throws std::runtime_error if command buffer recording fails.
     */
    void recordCommandBuffer(uint32_t frameIndex, uint32_t imageIndex) {
        VkCommandBuffer commandBuffer = commandBufferManager.getCommandBuffer(frameIndex);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error(std::string("failed to begin recording command buffer!") +
                                     std::to_string(frameIndex));
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[imageIndex];

        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChain.getSwapChainExtent();

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChain.getSwapChainExtent().width;
        viewport.height = (float) swapChain.getSwapChainExtent().height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChain.getSwapChainExtent();
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkBuffer vertexBuffers[] = { bufferManager.getVertexBuffer() };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, bufferManager.getIndexBuffer(), 0, getVulkanIndexType());

        vkCmdBindDescriptorSets(
            commandBuffer, 
            VK_PIPELINE_BIND_POINT_GRAPHICS, 
            pipelineLayout, 
            0, 
            1, 
            &descriptorManager.getDescriptorSet(frameIndex), 
            0, 
            nullptr
        );

        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the Vulkan pipeline layout.
     * 
     * @return The Vulkan pipeline layout.
     * @throws std::runtime_error if the pipeline layout is not initialized.
     */
    const VkPipelineLayout& getPipelineLayout() const {
        if (pipelineLayout == VK_NULL_HANDLE)
            throw std::runtime_error("Graphics pipeline layout is not initialized!");
        return pipelineLayout;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the Vulkan graphics pipeline.
     * 
     * @return The Vulkan graphics pipeline.
     * @throws std::runtime_error if the graphics pipeline is not initialized.
     */
    const VkPipeline& getPipeline() const {
        if (graphicsPipeline == VK_NULL_HANDLE)
            throw std::runtime_error("Graphics pipeline is not initialized!");
        return graphicsPipeline;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the Vulkan render pass used by this pipeline.
     * 
     * @return The Vulkan render pass.
     * @throws std::runtime_error if the render pass is not initialized.
     */
    const VkRenderPass& getRenderPass() const {
        if (renderPass == VK_NULL_HANDLE)
            throw std::runtime_error("Render pass is not initialized!");
        return renderPass;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the vector of framebuffers.
     * 
     * @return A vector of Vulkan framebuffers for each swap chain image.
     * @throws std::runtime_error if framebuffers are not populated.
     */
    const std::vector<VkFramebuffer>& getFrameBuffers() const {
        if (framebuffers.empty())
            std::runtime_error("Frame buffers vector is not populated!");
        return framebuffers;  
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves a framebuffer for a specific frame.
     * 
     * @param frameIndex The index of the framebuffer to retrieve.
     * @return The Vulkan framebuffer for the specified frame.
     * @throws std::runtime_error if framebuffers are empty or the frame index is out of bounds.
     */
    const VkFramebuffer& getFrameBuffer(uint32_t frameIndex) const {
        if (framebuffers.empty())
            std::runtime_error("Frame buffers vector is not populated!");
        if (frameIndex > size(framebuffers))
            std::runtime_error("Frame index is out of bounds!");
        return framebuffers[frameIndex];
    }
// ================================================================================
private:
    VkDevice device;                          /**< Vulkan logical device handle. */
    SwapChain& swapChain;                     /**< Reference to the swap chain. */
    CommandBufferManager<IndexType>& commandBufferManager;/**< Reference to the command buffer manager. */
    BufferManager<VertexType, IndexType>& bufferManager;             /**< Reference to the buffer manager. */
    DescriptorManager& descriptorManager;     /**< Reference to the descriptor manager. */
    const std::vector<IndexType>& indices;            /**< Index data for rendering. */
    VulkanPhysicalDevice& physicalDevice;          /**< Vulkan physical device object. */
    std::string vertFile;                     /**< Vertices Shader File. */ 
    std::string fragFile;                     /**< Fragmentation Shader File. */
    DepthManager& depthManager;               /**< Reference to DepthManager instance. */

    VkPipelineLayout pipelineLayout;          /**< The Vulkan pipeline layout. */
    VkPipeline graphicsPipeline;              /**< The Vulkan graphics pipeline. */
    VkRenderPass renderPass;                  /**< The Vulkan render pass. */
    std::vector<VkFramebuffer> framebuffers;  /**< Framebuffers for each swap chain image. */ 
// --------------------------------------------------------------------------------

    /**
     * @brief Creates a Vulkan shader module.
     * 
     * Loads shader code from a file and creates a shader module for the pipeline.
     * 
     * @param code The SPIR-V bytecode of the shader.
     * @return The created shader module.
     * @throws std::runtime_error if shader module creation fails.
     */
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Reads shader bytecode from a file.
     * 
     * Loads SPIR-V shader bytecode from the specified file path.
     * 
     * @param filename The path to the shader file.
     * @return The bytecode as a vector of chars.
     * @throws std::runtime_error if the file cannot be opened.
     */
    std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Current working directory: " << std::filesystem::current_path() << std::endl;
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Finds a suitable memory type for a Vulkan resource.
     * 
     * Searches for a memory type that matches the required properties.
     * 
     * @param typeFilter Bitmask of memory types to search.
     * @param properties Required memory properties.
     * @return The index of the suitable memory type.
     * @throws std::runtime_error if no suitable memory type is found.
     */
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice.getDevice(), &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Creates the Vulkan graphics pipeline.
     * 
     * Sets up shader stages, vertex input, rasterization, depth, and color blend states.
     * Compiles and links all stages into the graphics pipeline.
     * 
     * @throws std::runtime_error if pipeline creation fails.
     */
    void createGraphicsPipeline() {

        auto vertShaderCode = readFile(vertFile);
        auto fragShaderCode = readFile(fragFile);

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = VertexType::getBindingDescription();
        auto attributeDescriptions = VertexType::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;  // Set to 1 since you have one descriptor set layout
        pipelineLayoutInfo.pSetLayouts = &descriptorManager.getDescriptorSetLayout();  // Pass the descriptor set layout here
        pipelineLayoutInfo.pushConstantRangeCount = 0;  // No push constants for now

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Creates a Vulkan render pass.
     * 
     * Configures color and depth attachments for the render pass and sets up dependencies
     * for synchronization.
     * 
     * @param swapChainImageFormat Format of the swap chain images.
     * @throws std::runtime_error if render pass creation fails.
     */
    void createRenderPass(VkFormat swapChainImageFormat) {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depthManager.findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        // Correct the attachments array and count
        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());  // Set to 2
        renderPassInfo.pAttachments = attachments.data();  // Point to both color and depth attachments
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }
// --------------------------------------------------------------------------------

    /**
     * @brief Retrieves the appropriate Vulkan index type for the current IndexType.
     * 
     * Determines the Vulkan-compatible index type for the template parameter IndexType.
     * This supports uint8_t (with extension), uint16_t, and uint32_t types.
     * 
     * @return The Vulkan index type corresponding to IndexType.
     * @throws std::runtime_error if IndexType is unsupported or if uint8_t is used without extension support.
     */
    VkIndexType getVulkanIndexType() const {
        if constexpr (std::is_same<IndexType, uint32_t>::value) {
            return VK_INDEX_TYPE_UINT32;
        } else if constexpr (std::is_same<IndexType, uint16_t>::value) {
            return VK_INDEX_TYPE_UINT16;
        } else if constexpr (std::is_same<IndexType, uint8_t>::value) {
            // Check if the VK_EXT_index_type_uint8 extension is available for older Vulkan versions
            if (!physicalDevice.isExtensionSupported(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME)) {
                throw std::runtime_error("8-bit index type is not supported on this device.");
            }
            return VK_INDEX_TYPE_UINT8_EXT; // Works in both Vulkan 1.3+ and with the VK_EXT_index_type_uint8 extension
        } else {
            throw std::runtime_error("Unsupported index type");
        }
    }
};
// ================================================================================
// ================================================================================

template <typename VertexType, typename IndexType>
class RenderStrategy {
public:
    virtual ~RenderStrategy() = default;
// --------------------------------------------------------------------------------

    virtual bool beginFrame(uint32_t& frameIndex) = 0;
// --------------------------------------------------------------------------------
    
    virtual void recreateSwapChain() = 0;
// -------------------------------------------------------------------------------- 

    virtual void recordFrameCommands(uint32_t frameIndex, uint32_t imageIndex) = 0;
// --------------------------------------------------------------------------------

    virtual void endFrame(uint32_t frameIndex, uint32_t imageIndex, bool framebufferResized) = 0;
// --------------------------------------------------------------------------------

    virtual void updateUniformBuffer(uint32_t frameIndex, float zoomLevel) = 0;
// --------------------------------------------------------------------------------

    virtual uint32_t getCurrentFrame() const = 0;
// --------------------------------------------------------------------------------

 //   virtual uint32_t getImageIndex() const 0;
};
// ================================================================================
// ================================================================================

template <typename VertexType, typename IndexType>
class BasicRenderStrategy : public RenderStrategy<VertexType, IndexType> {
public:
    BasicRenderStrategy(CommandBufferManager<IndexType>& commandBufferManager,
                        GraphicsPipeline<VertexType, IndexType>& graphicsPipeline,
                        VulkanLogicalDevice& vulkanLogicalDevice,
                        DepthManager& depthManager,
                        VkQueue graphicsQueue,
                        VkQueue presentQueue,
                        SwapChain& swapChain,
                        BufferManager<VertexType, IndexType>& bufferManager)
        : commandBufferManager(commandBufferManager),
          graphicsPipeline(graphicsPipeline),
          vulkanLogicalDevice(vulkanLogicalDevice),
          depthManager(depthManager),
          graphicsQueue(graphicsQueue),
          presentQueue(presentQueue),
          swapChain(swapChain),
          bufferManager(bufferManager){}
// --------------------------------------------------------------------------------

    bool beginFrame(uint32_t& imageIndex) override {
        VkDevice device = vulkanLogicalDevice.getDevice();
        uint32_t frameIndex = currentFrame;

        // Wait for the frame to be finished 
        commandBufferManager.waitForFences(frameIndex);
        commandBufferManager.resetFences(frameIndex);

        VkResult result = vkAcquireNextImageKHR(device, swapChain.getSwapChain(), UINT64_MAX,
                                                commandBufferManager.getImageAvailableSemaphore(frameIndex),
                                                VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return false;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swap chain image!");
        }
        return true;
    }
// --------------------------------------------------------------------------------

    void recreateSwapChain() override {
        vkDeviceWaitIdle(vulkanLogicalDevice.getDevice());

        // Clean up existing swap chain and related resources
        graphicsPipeline.destroyFramebuffers();
        swapChain.cleanupSwapChain();

        // Recreate the swap chain and dependent resources
        swapChain.recreateSwapChain();

        // Recreate depth resources with the new extent
        depthManager.recreateDepthResources(swapChain.getSwapChainExtent());

        // Recreate the framebuffers using the new swap chain image views
        graphicsPipeline.createFrameBuffers(swapChain.getSwapChainImageViews(), swapChain.getSwapChainExtent());

        // Free existing command buffers
        VkCommandPool commandPool = commandBufferManager.getCommandPool();
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkCommandBuffer cmdBuffer = commandBufferManager.getCommandBuffer(i);
            vkFreeCommandBuffers(vulkanLogicalDevice.getDevice(), commandPool, 1, &cmdBuffer);
        }

        // Recreate the command buffers
        commandBufferManager.createCommandBuffers(); 
    }
// --------------------------------------------------------------------------------

    void recordFrameCommands(uint32_t frameIndex, uint32_t imageIndex) override {
        VkCommandBuffer cmdBuffer = commandBufferManager.getCommandBuffer(frameIndex);
        vkResetCommandBuffer(cmdBuffer, 0);
        graphicsPipeline.recordCommandBuffer(frameIndex, imageIndex); 
    }
// --------------------------------------------------------------------------------

    void endFrame(uint32_t frameIndex, uint32_t imageIndex, bool framebufferResized) override {
        VkCommandBuffer cmdBuffer = commandBufferManager.getCommandBuffer(frameIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {commandBufferManager.getImageAvailableSemaphore(frameIndex)};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        VkSemaphore signalSemaphores[] = {commandBufferManager.getRenderFinishedSemaphore(frameIndex)};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, commandBufferManager.getInFlightFence(frameIndex)) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain.getSwapChain()};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
// --------------------------------------------------------------------------------

    void updateUniformBuffer(uint32_t frameIndex, float zoomLevel) {
        UniformBufferObject ubo{};

        // Set the model matrix to the identity matrix (no rotation or scaling)
        ubo.model = glm::mat4(1.0f);

        // Set the view matrix to look from an isometric angle
        ubo.view = glm::lookAt(
            glm::vec3(2.0f, 2.0f, 2.0f),  // Position the camera diagonally above and to the side
            glm::vec3(0.0f, 0.0f, 0.0f),  // Look at the origin (center of the scene)
            glm::vec3(0.0f, 0.0f, 1.0f)   // Up direction is along the Z-axis
        );

        // Set the projection matrix with adjustable zoom level for an isometric perspective
        float fov = glm::radians(45.0f) / zoomLevel;  // Adjust the field of view based on zoom level
        ubo.proj = glm::perspective(fov,
                                    swapChain.getSwapChainExtent().width / (float)swapChain.getSwapChainExtent().height,
                                    0.1f, 10.0f);

        // Invert the Y-axis for Vulkan's coordinate system
        ubo.proj[1][1] *= -1;

        // Copy data to the mapped uniform buffer for the current frame
        memcpy(bufferManager.getUniformBuffersMapped()[frameIndex], &ubo, sizeof(ubo));
    }

    // This version of updateUniformBuffer rotates the objects
    // void updateUniformBuffer(uint32_t currentImage, float zoomLevel) override {
    //     static auto startTime = std::chrono::high_resolution_clock::now();
    //     auto currentTime = std::chrono::high_resolution_clock::now();
    //     float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    //
    //     UniformBufferObject ubo{};
    //     ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    //     ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    //     float fov = glm::radians(45.0f) / zoomLevel; // Adjust FOV with zoom level
    //     ubo.proj = glm::perspective(fov, swapChain.getSwapChainExtent().width / (float)swapChain.getSwapChainExtent().height, 0.1f, 10.0f);
    //     ubo.proj[1][1] *= -1; // Invert Y-axis for Vulkan
    //
    //     memcpy(bufferManager.getUniformBuffersMapped()[currentImage], &ubo, sizeof(ubo));
    // }
// --------------------------------------------------------------------------------

    uint32_t getCurrentFrame() const override {
        return currentFrame;
    }
// ================================================================================
private:
    CommandBufferManager<IndexType>& commandBufferManager;
    GraphicsPipeline<VertexType, IndexType>& graphicsPipeline;
    VulkanLogicalDevice& vulkanLogicalDevice;
    DepthManager& depthManager;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    SwapChain& swapChain;
    BufferManager<VertexType, IndexType>& bufferManager;

    uint32_t currentFrame = 0;
};
// ================================================================================
// ================================================================================
#endif /* graphics_HPP */
// ================================================================================
// ================================================================================
// eof
