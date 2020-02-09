#include <cuda_runtime.h>
#include <CUDASandbox/gpu_assert.hpp>
#include <CUDASandbox/rnd.hpp>
#include <algorithm>

#include "kernels/reduce.cuh"

metric reduce_cpu(const std::vector<int>& data)
{
	metric metric(data.size());
	auto result = 0;

	metric.start(metric_type::CALCULATION);
	for(const auto& val : data)
	{
		result += val;
	}
	metric.stop(metric_type::CALCULATION);

	printf("CPU Result: %d\n", result);
	return metric;
}

int main()
{
	const auto block_size = 512;
	std::vector<int> data(1 << 20);

	std::generate(data.begin(), data.end(), []()
	{
		return rnd::random<int>(-10000, 10000);
	});
	
	const auto cpu_metrics = reduce_cpu(data);
	const auto gpu_metrics = reduce_gpu::reduce_neighbored(data, block_size);

	cpu_metrics.serialize("CPU metrics");
	gpu_metrics.serialize("GPU metrics");
	
	GPU_ERR_CHECK(cudaDeviceReset());
    return 0;
}
