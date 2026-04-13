#pragma once

#include "framework/data_consumer.h"

#include <cstdint>
#include <map>
#include <string>

class ResultConsumer : public GryFlux::DataConsumer {
public:
    explicit ResultConsumer(const std::string& output_dir);
    ~ResultConsumer() override = default;

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

private:
    void WriteSequentialPacket(class FusionDataPacket* packet);

    std::string output_dir_;
    uint64_t expected_packet_idx_ = 0;
    std::map<uint64_t, std::unique_ptr<GryFlux::DataPacket>> reorder_buffer_;
};
