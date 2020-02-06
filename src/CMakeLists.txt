project(CUDASandbox)

set(CUDASB_ROOT ${PROJECT_SOURCE_DIR}/CUDASandbox)

set (CUDASB_SOURCES
    ${CUDASB_ROOT}/core.h
    ${CUDASB_ROOT}/gpu_assert.h
    ${CUDASB_ROOT}/gpu_assert.cpp
    ${CUDASB_ROOT}/rnd.h
    ${CUDASB_ROOT}/rnd.cpp
)

add_library(${PROJECT_NAME} SHARED ${CUDASB_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})

# set_target_properties(${PROJECT_NAME} 
#     PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#     POSITION_INDEPENDENT_CODE ON
# )

if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CUDASB_PLATFORM_WINDOWS=1)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDASB_BUILD_DLL=1)
endif()

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})