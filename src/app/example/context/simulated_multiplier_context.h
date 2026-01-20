#pragma once

#include "framework/context.h"

#include <atomic>

class SimulatedMultiplierContext : public GryFlux::Context
{
public:
    explicit SimulatedMultiplierContext(int deviceId) : deviceId_(deviceId) {}

    int getDeviceId() const { return deviceId_; }

    float mul(float a, float b)
    {
        opCount_.fetch_add(1, std::memory_order_relaxed);
        return a * b;
    }

    uint64_t getOpCount() const { return opCount_.load(std::memory_order_relaxed); }

private:
    int deviceId_ = 0;
    std::atomic<uint64_t> opCount_{0};
};
