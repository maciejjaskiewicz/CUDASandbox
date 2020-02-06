#pragma once
#include <cstdint>
#include <vector>

namespace reduce_gpu
{
	template<typename T>
	void reduce_neighbored(const std::vector<T>& data, uint16_t block_size);
}

template void reduce_gpu::reduce_neighbored(const std::vector<int>& data, uint16_t block_size);