#include "metric.hpp"
#include <cstdio>
#include <type_traits>

template<typename TResult>
metric<TResult>::metric(const std::size_t size) 
    : size_(size), result_{}
{ }

template<typename TResult>
void metric<TResult>::start(const metric_type metric_type)
{
    const auto now = std::chrono::system_clock::now();

    if(metric_type == metric_type::CALCULATION) 
    {
        calculations_start_ = now;
    }
    else
    {
        mem_transfer_start_ = now;
    }
}

template<typename TResult>
void metric<TResult>::stop(const metric_type metric_type)
{
    const auto now = std::chrono::system_clock::now();

    if(metric_type == metric_type::CALCULATION) 
    {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - calculations_start_);
        calculations_time_ = duration.count();
    }
    else
    {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - mem_transfer_start_);
        mem_transfer_time_ = duration.count();
    }
}

template<typename TResult>
void metric<TResult>::set_result(const TResult result)
{
    result_ = result;
}

template<typename TResult>
void metric<TResult>::serialize(const std::string& label) const
{
    printf("\033[1;33m[%s]\033[0m\n", label.c_str());
    printf(" - Problem size: n = %zu\n", size_);
    printf(" - Calculation time: %d ms\n", calculations_time_);
    printf(" - Memory transfer time: %d ms\n", mem_transfer_time_);
    printf(" - \033[1;31mTotal computation time: %d ms\033[0m\n", calculations_time_ + mem_transfer_time_);

    if(std::is_same<TResult, int>())
    {
        printf(" - Computation result: %d\n", result_);
    }
}

template class metric<int>;