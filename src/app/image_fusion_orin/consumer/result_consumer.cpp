#include "consumer/result_consumer.h"

#include "packet/fusion_data_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

ResultConsumer::ResultConsumer(const std::string& output_dir)
    : output_dir_(output_dir) {
    fs::create_directories(output_dir_);
    LOG.info("[ResultConsumer] Writing fused images to %s", output_dir_.c_str());
}

void ResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet) {
    auto* fusion_packet = static_cast<FusionDataPacket*>(packet.get());
    reorder_buffer_[fusion_packet->packet_idx] = std::move(packet);

    while (reorder_buffer_.count(expected_packet_idx_) > 0) {
        auto current_packet = std::move(reorder_buffer_[expected_packet_idx_]);
        reorder_buffer_.erase(expected_packet_idx_);
        WriteSequentialPacket(static_cast<FusionDataPacket*>(current_packet.get()));
        ++expected_packet_idx_;
    }
}

void ResultConsumer::WriteSequentialPacket(FusionDataPacket* packet) {
    if (packet->fused_result.empty()) {
        LOG.warning(
            "[ResultConsumer] Skip %s because fused output is empty",
            packet->filename.c_str());
        return;
    }

    const fs::path output_path = fs::path(output_dir_) / packet->filename;
    if (!cv::imwrite(output_path.string(), packet->fused_result)) {
        throw std::runtime_error("Failed to write fused image: " + output_path.string());
    }

    if (packet->packet_idx % 50 == 0) {
        LOG.info(
            "[ResultConsumer] Wrote packet %llu to %s",
            static_cast<unsigned long long>(packet->packet_idx),
            output_path.string().c_str());
    }
}
