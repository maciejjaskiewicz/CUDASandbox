#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <CUDASandbox/gpu_assert.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "reduce.cuh"

// high warp divergence!!!
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
void init_device(T*& d_data, T*& d_result, const std::vector<T>& data, 
	const std::size_t result_size, metric& metric)
{
	metric.start(metric_type::MEMORY_TRANSFER);

	const uint32_t data_byte_size = data.size() * sizeof(T);
	const uint32_t result_byte_size = result_size * sizeof(T);

	GPU_ERR_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), data_byte_size));
	GPU_ERR_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_result), result_byte_size));

	GPU_ERR_CHECK(cudaMemcpy(d_data, data.data(), data_byte_size, cudaMemcpyHostToDevice));
	GPU_ERR_CHECK(cudaMemset(d_result, 0, result_byte_size));

	metric.stop(metric_type::MEMORY_TRANSFER);
}

template<typename T>
T fetch_device_result(const T* d_result, const std::size_t size)
{
	const uint32_t result_byte_size = size * sizeof(T);

	std::vector<T> h_result(size);
	GPU_ERR_CHECK(cudaMemcpy(h_result.data(), d_result, result_byte_size, cudaMemcpyDeviceToHost));

	T gpu_result = 0;

	for(const auto& val : h_result)
	{
		gpu_result += val;
	}

	return gpu_result;
}

template<typename T>
metric reduce_gpu::reduce_neighbored(const std::vector<T>& data, const uint16_t block_size)
{
	T* d_data;
	T* d_result;
	metric metric(data.size());

	const dim3 block(block_size);
	const dim3 grid(data.size() / block_size);

	init_device(d_data, d_result, data, grid.x, metric);

	metric.start(metric_type::CALCULATION);
	reduce_neighbored_pairs<<<grid, block>>>(d_data, d_result, data.size());
	GPU_ERR_CHECK(cudaDeviceSynchronize());
	metric.stop(metric_type::CALCULATION);

	T gpu_result = fetch_device_result(d_result, grid.x);
	printf("GPU Result: %d\n", gpu_result);

	GPU_ERR_CHECK(cudaFree(d_data));
	GPU_ERR_CHECK(cudaFree(d_result));

	return metric;
}

// Explicit instantiations
template metric reduce_gpu::reduce_neighbored(const std::vector<int>& data, uint16_t block_size);