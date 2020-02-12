project(RgbToGray LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

set (RTG_SOURCES
    rgbToGray.cpp
)

set (RTG_CUDA_SOURCES
    kernels/rgbToGray.cuh
    kernels/rgbToGray.cu
)

set_source_files_properties(RTG_SOURCES PROPERTIES LANGUAGE CXX)
set_source_files_properties(RTG_CUDA_SOURCES PROPERTIES LANGUAGE CUDA)

# Assets
configure_file(assets/eiffel.jpg assets/eiffel.jpg COPYONLY)
configure_file(assets/island.jpg assets/island.jpg COPYONLY)

# Define executable
add_executable(${PROJECT_NAME} 
    ${RTG_SOURCES}
    ${RTG_CUDA_SOURCES}
)
set_target_properties(${PROJECT_NAME} 
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} CUDASandbox)