#include <cuda_runtime.h>
#include <CUDASandbox/gpu_assert.hpp>
#include <CUDASandbox/rnd.hpp>
#include <algorithm>

#include "kernels/reduce.cuh"

template <typename T>
metric_with_result<T> reduce_cpu(const std::vector<T>& data)
{
	metric_with_result<T> metric(data.size());
	T result = 0;

	metric.start(metric_type::CALCULATION);
	for(const auto& val : data)
	{
		result += val;
	}
	metric.stop(metric_type::CALCULATION);
	metric.set_result(result);

	return metric;
}

int main()
{
	const auto block_size = 256;
	std::vector<int> data(1 << 25);

	std::generate(data.begin(), data.end(), []()
	{
		return rnd::random<int>(-10000, 10000);
	});
	
	const auto cpu_metrics = reduce_cpu(data);
	const auto gpu_neighbored_metrics = reduce_gpu::reduce_neighbored(data, block_size);
	const auto gpu_neighbored_imp_metrics = reduce_gpu::reduce_neighbored_imp(data, block_size);
	const auto gpu_interleaved_metrics = reduce_gpu::reduce_interleaved(data, block_size);
	const auto gpu_unrolling_2blocks_metrics = reduce_gpu::reduce_unrolling_blocks(data, block_size, 2);
	const auto gpu_unrolling_4blocks_metrics = reduce_gpu::reduce_unrolling_blocks(data, block_size, 4);
	const auto gpu_unrolling_warps_metrics = reduce_gpu::reduce_unrolling_warps(data, block_size);
	const auto gpu_complete_unrolling_metrics = reduce_gpu::reduce_complete_unrolling(data, block_size);

	cpu_metrics.serialize("CPU metrics");
	gpu_neighbored_metrics.serialize("GPU - Neighbored pairs");
	gpu_neighbored_imp_metrics.serialize("GPU - Improved neighbored pairs");
	gpu_interleaved_metrics.serialize("GPU - Interleaved pairs");
	gpu_unrolling_2blocks_metrics.serialize("GPU - 2 blocks unrolling");
	gpu_unrolling_4blocks_metrics.serialize("GPU - 4 blocks unrolling");
	gpu_unrolling_warps_metrics.serialize("GPU - Warps unrolling");
	gpu_complete_unrolling_metrics.serialize("GPU - Complete unrolling");
	
	GPU_ERR_CHECK(cudaDeviceReset());
    return 0;
}
