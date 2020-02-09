#pragma once
#include <cstdint>
#include <vector>

#include <CUDASandbox/metric.hpp>

namespace reduce_gpu
{
	template<typename T>
	metric reduce_neighbored(const std::vector<T>& data, uint16_t block_size);
}