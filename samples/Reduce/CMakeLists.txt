project(Reduce LANGUAGES CXX CUDA)

set (REDUCE_SOURCES
    reduce.cpp
)

set (REDUCE_CUDA_SOURCES
    kernels/reduce.cu
    kernels/reduce.cuh
)

set_source_files_properties(REDUCE_CUDA_SOURCES PROPERTIES LANGUAGE CUDA)

# Define executable
add_executable(${PROJECT_NAME} 
    ${REDUCE_SOURCES}
    ${REDUCE_CUDA_SOURCES}
)
set_target_properties(${PROJECT_NAME} 
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PUBLIC CUDASandbox)