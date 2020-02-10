#pragma once
#include "core.hpp"

#include <chrono>
#include <string>

enum CSB_API metric_type
{
    MEMORY_TRANSFER,
    CALCULATION
};

template<typename TResult>
class CSB_API metric
{
public:
    metric(std::size_t size);

    void start(const metric_type metric_type);
    void stop(const metric_type metric_type);
    void set_result(const TResult result);

    void serialize(const std::string& label) const;

private:
    std::size_t size_;
    TResult result_;
    int mem_transfer_time_{};
    int calculations_time_{};
    std::chrono::system_clock::time_point mem_transfer_start_;
    std::chrono::system_clock::time_point calculations_start_;
};