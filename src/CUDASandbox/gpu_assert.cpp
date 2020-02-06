#include "gpu_assert.h"

#include <cstdio>
#include <iostream>

inline void CUDASB_API gpu_assert(const cudaError_t& code, const char *file, const int line, const bool abort)
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