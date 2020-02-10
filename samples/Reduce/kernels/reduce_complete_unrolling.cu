#include <device_launch_parameters.h>

#include "reduce.cuh"
#include "reduce_helpers.cuh"

template<uint16_t blocks_size>
__global__ void reduce_complete_unrolling_kernel(int* input, int* result, uint32_t size)
{
	const int tid = threadIdx.x;
    int* input_local = input + blockDim.x * blockIdx.x;

    if (blocks_size >= 1024) { if (tid < 512) { input_local[tid] += input_local[tid + 512]; } __syncthreads(); }
    if (blocks_size >= 512) { if (tid < 256) { input_local[tid] += input_local[tid + 256]; } __syncthreads(); }
    if (blocks_size >= 256) { if (tid < 128) { input_local[tid] += input_local[tid + 128]; } __syncthreads(); }
    if (blocks_size >= 128) { if (tid < 64) { input_local[tid] += input_local[tid + 64]; } __syncthreads(); }

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
metric<T> reduce_gpu::reduce_complete_unrolling(const std::vector<T>& data, const uint16_t block_size)
{
	T* d_data;
	T* d_result;
	metric<T> metric(data.size());

	const dim3 block(block_size);
	const dim3 grid(data.size() / block_size);

	init_device(d_data, d_result, data, grid.x, metric);

	metric.start(metric_type::CALCULATION);
    switch(block_size)
    {
        case 1024: reduce_complete_unrolling_kernel<1024><<<grid, block>>>(d_data, d_result, data.size()); break;
        case 512: reduce_complete_unrolling_kernel<512><<<grid, block>>>(d_data, d_result, data.size()); break;
        case 256: reduce_complete_unrolling_kernel<256><<<grid, block>>>(d_data, d_result, data.size()); break;
        case 128: reduce_complete_unrolling_kernel<128><<<grid, block>>>(d_data, d_result, data.size()); break;
    }
	
	GPU_ERR_CHECK(cudaDeviceSynchronize());
	metric.stop(metric_type::CALCULATION);

	T gpu_result = fetch_device_result(d_result, grid.x);
	metric.set_result(gpu_result);

	GPU_ERR_CHECK(cudaDeviceReset());

	return metric;
}

// Explicit instantiations
template metric<int> reduce_gpu::reduce_complete_unrolling(const std::vector<int>& data, const uint16_t block_size);