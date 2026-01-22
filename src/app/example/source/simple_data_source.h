/*************************************************************************************************************************
 * Copyright 2025 GKGgood
 *
 * GryFlux Framework - Simple Data Source (Example)
 *************************************************************************************************************************/
#pragma once

#include "framework/data_source.h"
#include "packet/simple_data_packet.h"

#include <chrono>
#include <thread>

/**
 * @brief 简单数据源 - 示例实现
 *
 * 产生指定数量的简单数据包，每个包只设置 id。
 * value 会在 InputNode 中初始化。
 */
class SimpleDataSource : public GryFlux::DataSource
{
public:
    /**
     * @brief 构造函数
     *
     * @param totalPackets 要产生的数据包总数
     */
    explicit SimpleDataSource(size_t totalPackets, size_t producerTime)
        : totalPackets_(totalPackets), producedCount_(0), producerTime_(producerTime)
    {
        setHasMore(totalPackets_ > 0);
    }

    std::unique_ptr<GryFlux::DataPacket> produce() override
    {
        if (producedCount_ >= totalPackets_)
        {
            setHasMore(false);
            return nullptr;
        }
        auto packet = std::make_unique<SimpleDataPacket>();
        packet->id = static_cast<int>(producedCount_);
        // value 会在 InputNode 中初始化为 id

        std::this_thread::sleep_for(std::chrono::milliseconds(producerTime_)); // 模拟数据准备延时
        producedCount_++;
        setHasMore(producedCount_ < totalPackets_);
        return packet;
    }

private:
    size_t totalPackets_;
    size_t producedCount_;
    size_t producerTime_;
};
