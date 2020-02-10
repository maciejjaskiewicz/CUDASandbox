project(Reduce LANGUAGES CXX CUDA)

set (REDUCE_SOURCES
    reduce.cpp
)

set (REDUCE_CUDA_SOURCES
    kernels/reduce.cuh
    kernels/reduce_helpers.cuh
    kernels/reduce_neighbored.cu
    kernels/reduce_neighbored_imp.cu
    kernels/reduce_interleaved.cu
    kernels/reduce_unrolling_blocks.cu
    kernels/reduce_unrolling_warps.cu
    kernels/reduce_complete_unrolling.cu
)

set_source_files_properties(REDUCE_SOURCES PROPERTIES LANGUAGE CXX)
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