#pragma once
#include <cstdint>
#include <vector>

#include <CUDASandbox/metric.hpp>

namespace reduce_gpu
{
	template<typename T>
	metric_with_result<T> reduce_neighbored(const std::vector<T>& data, uint16_t block_size);
	template<typename T>
	metric_with_result<T> reduce_neighbored_imp(const std::vector<T>& data, uint16_t block_size);
	template<typename T>
	metric_with_result<T> reduce_interleaved(const std::vector<T>& data, uint16_t block_size);
	template<typename T>
	metric_with_result<T> reduce_unrolling_blocks(const std::vector<T>& data, uint16_t block_size,
		const uint8_t blocks_to_unroll);
	template<typename T>
	metric_with_result<T> reduce_unrolling_warps(const std::vector<T>& data, uint16_t block_size);
	template<typename T>
	metric_with_result<T> reduce_complete_unrolling(const std::vector<T>& data, const uint16_t block_size);
}