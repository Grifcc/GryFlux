#pragma once

#include "framework/data_consumer.h"
#include <string>

class FusionDataConsumer : public GryFlux::DataConsumer {
public:
    explicit FusionDataConsumer(const std::string& saveDir);
    ~FusionDataConsumer() override = default;

    // 框架会自动调用此方法，传入处理完的数据包
    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

private:
    std::string saveDir_;
};