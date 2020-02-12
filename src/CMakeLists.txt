project(CUDASandbox LANGUAGES CXX)

set(CSB_ROOT ${PROJECT_SOURCE_DIR}/CUDASandbox)

set (CSB_SOURCES
    # ${CSB_ROOT}/core.hpp
    ${CSB_ROOT}/gpu_assert.hpp
    ${CSB_ROOT}/rnd.hpp
    ${CSB_ROOT}/rnd.cpp
    ${CSB_ROOT}/metric.hpp
    ${CSB_ROOT}/metric.cpp
)

add_library(${PROJECT_NAME} STATIC ${CSB_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
# set_target_properties(${PROJECT_NAME} 
#     PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#     POSITION_INDEPENDENT_CODE ON
# )

if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CSB_PLATFORM_WINDOWS=1)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CSB_BUILD_DLL=1)
endif()
if(APPLE)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CSB_PLATFORM_APPLE=1)
endif()
if(UNIX AND NOT APPLE)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CSB_PLATFORM_UNIX=1)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CSB_BUILD_DLL=1)
endif()

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})