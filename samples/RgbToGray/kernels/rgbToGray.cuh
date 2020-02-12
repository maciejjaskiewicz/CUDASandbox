#pragma once
#include <CUDASandbox/metric.hpp>
#include <cuda_runtime.h>

namespace gpu
{
	void rgb_to_gray(const uchar3* const rgb_img, uint8_t*& gray_img, std::size_t rows, 
		std::size_t cols, metric& metric);
}
