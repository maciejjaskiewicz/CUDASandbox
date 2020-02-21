#include "rgbToGray.cuh"
#include <CUDASandbox/gpu_assert.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void rgb_to_gray_kernel(const uchar3* const rgb, uint8_t* const gray, 
    const uint32_t rows, const uint32_t cols)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gid = iy * rows + ix;

    if(ix < rows && iy < cols)
    { 
        const int r = rgb[gid].x * .299F;
        const int g = rgb[gid].y * .587F;
        const int b = rgb[gid].z * .114F;
        
        gray[gid] = r + g + b;
    }
}

void gpu::rgb_to_gray(const uchar3* const rgb_img, uint8_t*& gray_img, std::size_t rows, 
    std::size_t cols, metric& metric)
{
    uchar3* d_rgb;
    uint8_t* d_gray;

    const auto pixels = rows * cols;
    const auto rgb_byte_size = pixels * sizeof(uchar3);
    const auto gray_byte_size = pixels * sizeof(uint8_t);

    metric.start(metric_type::HTD_MEMORY_TRANSFER);
    GPU_ERR_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rgb), rgb_byte_size));
    GPU_ERR_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gray), gray_byte_size));

    GPU_ERR_CHECK(cudaMemcpy(d_rgb, rgb_img, rgb_byte_size, cudaMemcpyHostToDevice));
    metric.stop(metric_type::HTD_MEMORY_TRANSFER);

    const dim3 block(128, 8);
    const dim3 grid((rows / block.x) + 1, (cols / block.y) + 1);

    metric.start(metric_type::CALCULATION);
    rgb_to_gray_kernel<<<grid, block>>>(d_rgb, d_gray, rows, cols);
    GPU_ERR_CHECK(cudaDeviceSynchronize());
    metric.stop(metric_type::CALCULATION);

    metric.start(metric_type::DTH_MEMORY_TRANSFER);

    GPU_ERR_CHECK(cudaMemcpy(gray_img, d_gray, gray_byte_size, cudaMemcpyDeviceToHost));
    metric.stop(metric_type::DTH_MEMORY_TRANSFER);
    
    GPU_ERR_CHECK(cudaDeviceReset());
}