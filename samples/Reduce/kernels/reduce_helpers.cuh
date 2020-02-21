#pragma once
#include <cuda_runtime.h>
#include <CUDASandbox/metric.hpp>
#include <CUDASandbox/gpu_assert.hpp>
#include <vector>

template<typename T>
void init_device(T*& d_data, T*& d_result, const std::vector<T>& data, 
    const std::size_t result_size, metric_with_result<T>& metric)
{
    metric.start(metric_type::HTD_MEMORY_TRANSFER);

    const uint32_t data_byte_size = data.size() * sizeof(T);
    const uint32_t result_byte_size = result_size * sizeof(T);

    GPU_ERR_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), data_byte_size));
    GPU_ERR_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_result), result_byte_size));

    GPU_ERR_CHECK(cudaMemcpy(d_data, data.data(), data_byte_size, cudaMemcpyHostToDevice));
    GPU_ERR_CHECK(cudaMemset(d_result, 0, result_byte_size));

    metric.stop(metric_type::HTD_MEMORY_TRANSFER);
}

template<typename T>
T fetch_device_result(const T* d_result, const std::size_t size)
{
    const uint32_t result_byte_size = size * sizeof(T);

    std::vector<T> h_result(size);
    GPU_ERR_CHECK(cudaMemcpy(h_result.data(), d_result, result_byte_size, cudaMemcpyDeviceToHost));

    T gpu_result = 0;

    for(const auto& val : h_result)
    {
        gpu_result += val;
    }

    return gpu_result;
}