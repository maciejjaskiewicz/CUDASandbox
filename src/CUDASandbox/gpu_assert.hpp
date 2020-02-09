#pragma once
#include <cuda_runtime.h>
#include <iostream>

#include "core.hpp"

#define GPU_ERR_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline CSB_API void gpu_assert(const cudaError_t& code, const char *file, int line, bool abort = true)
{
	if(code != cudaSuccess)
	{
		fprintf(stderr, "GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);

		if(abort)
		{
 			exit(code);
		}
	}
}