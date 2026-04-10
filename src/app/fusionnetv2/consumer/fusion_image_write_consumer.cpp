#include "consumer/fusion_image_write_consumer.h"

#include "packet/fusionnetv2_packet.h"
#include "utils/logger.h"

#include <opencv2/imgcodecs.hpp>

#include <stdexcept>

FusionImageWriteConsumer::FusionImageWriteConsumer(const std::string &outputDir)
    : outputDir_(outputDir)
{
    if (outputDir_.empty())
    {
        throw std::runtime_error("Output directory is empty");
    }

    fs::create_directories(outputDir_);
    LOG.info("FusionImageWriteConsumer output=%s", outputDir_.string().c_str());
}

void FusionImageWriteConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet)
{
    if (!packet)
    {
        return;
    }

    consumedCount_.fetch_add(1, std::memory_order_relaxed);

    auto &p = static_cast<FusionNetV2Packet &>(*packet);
    if (p.outputBgrU8.empty())
    {
        LOG.warning("FusionImageWriteConsumer got empty output for idx=%d", p.idx);
        return;
    }

    std::string filename = p.filename;
    if (filename.empty())
    {
        filename = "fusion_" + std::to_string(p.idx) + ".png";
    }

    const fs::path outputPath = outputDir_ / filename;
    if (!cv::imwrite(outputPath.string(), p.outputBgrU8))
    {
        LOG.error("Failed to write fusion result: %s", outputPath.string().c_str());
        return;
    }

    writtenCount_.fetch_add(1, std::memory_order_relaxed);
}
