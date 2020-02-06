#pragma once
#include <cuda_runtime.h>

#include "core.h"

#define GPU_ERR_CHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void CUDASB_API gpu_assert(const cudaError_t& code, const char *file, int line, bool abort = true);