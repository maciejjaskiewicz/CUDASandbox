#pragma once
#include "core.hpp"

#include <chrono>
#include <string>

enum CSB_API metric_type
{
    MEMORY_TRANSFER,
    CALCULATION
};

class CSB_API metric
{
public:
    metric(std::size_t size);

    void start(const metric_type metric_type);
    void stop(const metric_type metric_type);

    void serialize(const std::string& label) const;

private:
    std::size_t size_;
    double mem_transfer_time_{};
    double calculations_time_{};
    std::chrono::system_clock::time_point mem_transfer_start_;
    std::chrono::system_clock::time_point calculations_start_;
};