#include "metric.hpp"
#include <cstdio>
#include <type_traits>

metric::metric(const std::size_t size)
    : size_(size)
{ }

template<typename TResult>
metric_with_result<TResult>::metric_with_result(const std::size_t size)
    : metric(size), result_{}
{ }

void metric::start(const metric_type metric_type)
{
    const auto now = std::chrono::system_clock::now();

    switch (metric_type)
    {
        case metric_type::CALCULATION:
            calculations_start_ = now;
            break;
        case metric_type::HTD_MEMORY_TRANSFER:
            htd_mem_transfer_start_ = now;
            break;
        case metric_type::DTH_MEMORY_TRANSFER:
            dth_mem_transfer_start_ = now;
            break;
    }
}

void metric::stop(const metric_type metric_type)
{
    const auto now = std::chrono::system_clock::now();

    switch (metric_type)
    {
        case metric_type::CALCULATION:
        {
            const auto c_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - calculations_start_);
            calculations_time_ = c_duration.count();
            break;
        }
        case metric_type::HTD_MEMORY_TRANSFER:
        {
            const auto htd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - htd_mem_transfer_start_);
            htd_mem_transfer_time_ = htd_duration.count();
            break;
        }
        case metric_type::DTH_MEMORY_TRANSFER:
        {
            const auto dth_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - dth_mem_transfer_start_);
            dth_mem_transfer_time_ = dth_duration.count();
            break;
        }
    }
}

template<typename TResult>
void metric_with_result<TResult>::set_result(const TResult result)
{
    result_ = result;
}

void metric::serialize(const std::string& label) const
{
    printf("\033[1;33m[%s]\033[0m\n", label.c_str());
    printf(" - Problem size: n = %zu\n", size_);
    printf(" - Calculation time: %d ms\n", calculations_time_);
    printf(" - Host to device memory transfer time: %d ms\n", htd_mem_transfer_time_);
    printf(" - Device to host memory transfer time: %d ms\n", dth_mem_transfer_time_);
    printf(" - \033[1;31mTotal computation time: %d ms\033[0m\n", 
        calculations_time_ + htd_mem_transfer_time_ + dth_mem_transfer_time_);
}

template<typename TResult>
void metric_with_result<TResult>::serialize(const std::string& label) const
{
    printf("\033[1;33m[%s]\033[0m\n", label.c_str());
    printf(" - Problem size: n = %zu\n", size_);
    printf(" - Calculation time: %d ms\n", calculations_time_);
    printf(" - Host to device memory transfer time: %d ms\n", htd_mem_transfer_time_);
    printf(" - Device to host memory transfer time: %d ms\n", dth_mem_transfer_time_);
    printf(" - \033[1;31mTotal computation time: %d ms\033[0m\n",
        calculations_time_ + htd_mem_transfer_time_ + dth_mem_transfer_time_);

    if (std::is_same<TResult, int>())
    {
        printf(" - Computation result: %d\n", result_);
    }
}

template class metric_with_result<int>;