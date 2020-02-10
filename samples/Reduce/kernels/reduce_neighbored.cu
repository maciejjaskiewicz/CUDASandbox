#include <device_launch_parameters.h>

#include "reduce.cuh"
#include "reduce_helpers.cuh"

// reduced warp divergence
__global__ void reduce_neighbored_pairs(int* input, int* result, uint32_t size)
{
	const int tid = threadIdx.x;
	const int gid = blockDim.x * blockIdx.x + tid;

	if(gid > size) return;
	
	for(auto offset = 1; offset <= blockDim.x / 2; offset *= 2)
	{
		if(tid % (2 * offset) == 0)
		{
			input[gid] += input[gid + offset];
		}

		__syncthreads();
	}

	if(tid == 0)
	{
		result[blockIdx.x] = input[gid];
	}
}

template<typename T>
metric<T> reduce_gpu::reduce_neighbored(const std::vector<T>& data, const uint16_t block_size)
{
	T* d_data;
	T* d_result;
	metric<T> metric(data.size());

	const dim3 block(block_size);
	const dim3 grid(data.size() / block_size);

	init_device(d_data, d_result, data, grid.x, metric);

	metric.start(metric_type::CALCULATION);
	reduce_neighbored_pairs<<<grid, block>>>(d_data, d_result, data.size());
	GPU_ERR_CHECK(cudaDeviceSynchronize());
	metric.stop(metric_type::CALCULATION);

	T gpu_result = fetch_device_result(d_result, grid.x);
	metric.set_result(gpu_result);

	GPU_ERR_CHECK(cudaDeviceReset());

	return metric;
}

// Explicit instantiations
template metric<int> reduce_gpu::reduce_neighbored(const std::vector<int>& data, uint16_t block_size);