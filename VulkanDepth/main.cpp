// ================================================================================
// ================================================================================
// - File:    main.cpp
// - Purpose: Describe the file purpose here
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    July 26, 2024
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here

#include "include/application.hpp"
#include <iostream>
#include <stdexcept>
#include <memory>
// ================================================================================
// ================================================================================ 


// Begin code
int main(int argc, const char * argv[]) {

    // Define vertices for two connected triangles  
    const std::vector<ThreeDVertex> vertices = {
        // First square, slightly offset along the x-axis and closer along the z-axis
        {{-0.4f, -0.4f, -0.25f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
        {{0.4f, -0.4f, -0.25f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{0.4f, 0.4f, -0.25f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
        {{-0.4f, 0.4f, -0.25f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

        // Second square, further along z-axis and offset along x-axis
        {{-0.5f + 0.5f, -0.5f, -0.75f}, {1.0f, 0.5f, 0.0f}, {1.0f, 0.0f}},
        {{0.5f + 0.5f, -0.5f, -0.75f}, {0.5f, 1.0f, 0.5f}, {0.0f, 0.0f}},
        {{0.5f + 0.5f, 0.5f, -0.75f}, {0.0f, 0.5f, 1.0f}, {0.0f, 1.0f}},
        {{-0.5f + 0.5f, 0.5f, -0.75f}, {0.5f, 1.0f, 1.0f}, {1.0f, 1.0f}}
    };
    const std::vector<uint16_t> indices = {
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4
    };
    
    // Call Application 
    try {
        VulkanApplication<ThreeDVertex, uint16_t> app = VulkanApplicationBuilder<ThreeDVertex, uint16_t>() 
            .setVertexInfo(vertices, indices)
            .setBaseShaderPaths("../../shaders/shader.vert.spv", 
                                "../../shaders/shader.frag.spv")
            .setTexturePath("../../../data/texture.jpg")
            .build(1050, 1200, "Vulkan Application", false);

        app.run();

    } catch(const std::exception& e) {
        std::cerr << e.what() <<  "\n";
        return EXIT_FAILURE;
    }
	return EXIT_SUCCESS;
}
// ================================================================================
// ================================================================================
// eof

