/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Simulated Tracker Context
 *************************************************************************************************************************/
#pragma once

#include "framework/context.h"
#include <atomic>
#include <chrono>
#include <thread>

/**
 * @brief Simulated Tracker Context
 *
 * 模拟“跨帧依赖”的跟踪器：全局只能有一个上下文实例被持有。
 * 通过 ResourcePool 注册为资源类型 "tracker"（仅 1 个实例），保证串行执行。
 */
class SimulatedTrackerContext : public GryFlux::Context
{
public:
    SimulatedTrackerContext() = default;
    ~SimulatedTrackerContext() override = default;

    void updateFrame(int frameId)
    {
        lastFrameId_.store(frameId, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    int getLastFrameId() const
    {
        return lastFrameId_.load(std::memory_order_acquire);
    }

private:
    std::atomic<int> lastFrameId_{-1};
};
