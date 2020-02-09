#include "metric.hpp"
#include <cstdio>

metric::metric(std::size_t size) 
    : size_(size)
{ }

void metric::start(const metric_type metric_type)
{
    const auto now = std::chrono::high_resolution_clock::now();

    if(metric_type == metric_type::CALCULATION) 
    {
        calculations_start_ = now;
    }
    else
    {
        mem_transfer_start_ = now;
    }
}

void metric::stop(const metric_type metric_type)
{
    const auto now = std::chrono::high_resolution_clock::now();

    if(metric_type == metric_type::CALCULATION) 
    {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - calculations_start_);
        calculations_time_ = duration.count() / 1000.0;
    }
    else
    {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - mem_transfer_start_);
        mem_transfer_time_ = duration.count() / 1000.0;
    }
}

void metric::serialize(const std::string& label) const
{
    printf("[%s]\n", label.c_str());
    printf(" - Problem size: n = %d\n", size_);
    printf(" - Calculation time: %.10lf\n", calculations_time_);
    printf(" - Memory transfer time: %.10lf\n", mem_transfer_time_);
    printf(" - Total computation time: %.10lf\n", calculations_time_ + mem_transfer_time_);
}