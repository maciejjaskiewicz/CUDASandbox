#include <cuda_runtime.h>
#include <CUDASandbox/gpu_assert.h>
#include <CUDASandbox/rnd.h>
#include <algorithm>

#include "kernels/reduce.cuh"

int reduce_cpu(const std::vector<int>& data)
{
	auto result = 0;

	for(const auto& val : data)
	{
		result += val;
	}

	return result;
}

int main()
{
	const auto block_size = 512;
	std::vector<int> data(1 << 26);

	std::generate(data.begin(), data.end(), []()
	{
		return rnd::random<int>(-10000, 10000);
	});
	
	reduce_gpu::reduce_neighbored(data, block_size);
	const auto cpu_result = reduce_cpu(data);

	printf("CPU Result: %d\n", cpu_result);
	
	GPU_ERR_CHECK(cudaDeviceReset());
    return 0;
}
