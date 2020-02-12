#include <device_launch_parameters.h>

#include "reduce.cuh"
#include "reduce_helpers.cuh"

__global__ void reduce_unrolling_2blocks(int* input, int* result, uint32_t size)
{
	const int tid = threadIdx.x;
    const int block_offset = blockDim.x * blockIdx.x * 2; // 2 blocks unrolling
	const int index = block_offset + tid;

    int* input_local = input + block_offset;

    if((index + blockDim.x) < size)
    {
        input[index] += input[index + blockDim.x];
    }

    __syncthreads();
	
	for(auto offset = blockDim.x / 2; offset > 0; offset /= 2)
	{
		if(tid < offset)
        {
            input_local[tid] += input_local[tid + offset];
        }

		__syncthreads();
	}

	if(tid == 0)
	{
		result[blockIdx.x] = input_local[0];
	}
}

__global__ void reduce_unrolling_4blocks(int* input, int* result, uint32_t size)
{
	const int tid = threadIdx.x;
    const int block_offset = blockDim.x * blockIdx.x * 4; // 4 blocks unrolling
	const int index = block_offset + tid;

    int* input_local = input + block_offset;

    if((index + 3 * blockDim.x) < size)
    {
        int u1 = input[index];
        int u2 = input[index + blockDim.x];
        int u3 = input[index + 2 * blockDim.x];
        int u4 = input[index + 3 * blockDim.x];
        input[index] = u1 + u2 + u3 + u4;
    }

    __syncthreads();
	
	for(auto offset = blockDim.x / 2; offset > 0; offset /= 2)
	{
		if(tid < offset)
        {
            input_local[tid] += input_local[tid + offset];
        }

		__syncthreads();
	}

	if(tid == 0)
	{
		result[blockIdx.x] = input_local[0];
	}
}

template<typename T>
metric_with_result<T> reduce_gpu::reduce_unrolling_blocks(const std::vector<T>& data, const uint16_t block_size,
    const uint8_t blocks_to_unroll)
{
	T* d_data;
	T* d_result;
	metric_with_result<T> metric(data.size());

	const dim3 block(block_size);
	const dim3 grid((data.size() / block_size) / blocks_to_unroll);

	init_device(d_data, d_result, data, grid.x, metric);

	metric.start(metric_type::CALCULATION);
    if(blocks_to_unroll == 4)
    {
        reduce_unrolling_4blocks<<<grid, block>>>(d_data, d_result, data.size());
    }
	else
    {
        reduce_unrolling_2blocks<<<grid, block>>>(d_data, d_result, data.size());
    }
	GPU_ERR_CHECK(cudaDeviceSynchronize());
	metric.stop(metric_type::CALCULATION);

	T gpu_result = fetch_device_result(d_result, grid.x);
	metric.set_result(gpu_result);

	GPU_ERR_CHECK(cudaDeviceReset());

	return metric;
}

// Explicit instantiations
template metric_with_result<int> reduce_gpu::reduce_unrolling_blocks(const std::vector<int>& data, uint16_t block_size,
    const uint8_t blocks_to_unroll);