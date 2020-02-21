#include <device_launch_parameters.h>

#include "reduce.cuh"
#include "reduce_helpers.cuh"

__global__ void reduce_unrolling_warps_kernel(int* input, int* result, uint32_t size)
{
    const int tid = threadIdx.x;
    int* input_local = input + blockDim.x * blockIdx.x;
    
    for(auto offset = blockDim.x / 2; offset >= 64; offset /= 2)
    {
        if(tid < offset)
        {
            input_local[tid] += input_local[tid + offset];
        }

        __syncthreads();
    }

    if(tid < 32)
    {
        volatile int* vsmem = input_local;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if(tid == 0)
    {
        result[blockIdx.x] = input_local[0];
    }
}

template<typename T>
metric_with_result<T> reduce_gpu::reduce_unrolling_warps(const std::vector<T>& data, const uint16_t block_size)
{
    T* d_data;
    T* d_result;
    metric_with_result<T> metric(data.size());

    const dim3 block(block_size);
    const dim3 grid(data.size() / block_size);

    init_device(d_data, d_result, data, grid.x, metric);

    metric.start(metric_type::CALCULATION);
    reduce_unrolling_warps_kernel<<<grid, block>>>(d_data, d_result, data.size());
    GPU_ERR_CHECK(cudaDeviceSynchronize());
    metric.stop(metric_type::CALCULATION);

    T gpu_result = fetch_device_result(d_result, grid.x);
    metric.set_result(gpu_result);

    GPU_ERR_CHECK(cudaDeviceReset());

    return metric;
}

// Explicit instantiations
template metric_with_result<int> reduce_gpu::reduce_unrolling_warps(const std::vector<int>& data, uint16_t block_size);