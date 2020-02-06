#pragma once
#include "core.h"

namespace rnd 
{
    template<typename T>
    T CUDASB_API random(T min, T max);
}

template CUDASB_API int rnd::random<int>(int min, int max);
template CUDASB_API float rnd::random<float>(float min, float max);
template CUDASB_API double rnd::random<double>(double min, double max);