#pragma once
#include <chrono>
#include <string>

enum metric_type
{
    HTD_MEMORY_TRANSFER,
    DTH_MEMORY_TRANSFER,
    CALCULATION
};

class metric
{
public:
    metric(std::size_t size);

    void start(const metric_type metric_type);
    void stop(const metric_type metric_type);

    virtual void serialize(const std::string& label) const;

protected:
    std::size_t size_;
    int htd_mem_transfer_time_{};
    int dth_mem_transfer_time_{};
    int calculations_time_{};
    std::chrono::system_clock::time_point htd_mem_transfer_start_;
    std::chrono::system_clock::time_point dth_mem_transfer_start_;
    std::chrono::system_clock::time_point calculations_start_;
};

template<typename TResult>
class metric_with_result : public metric
{
public:
    metric_with_result(std::size_t size);
    void set_result(const TResult result);

    void serialize(const std::string& label) const override;

private:
    TResult result_;
};