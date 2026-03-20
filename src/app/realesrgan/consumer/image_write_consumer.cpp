#include "consumer/image_write_consumer.h"

#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <stdexcept>

ImageWriteConsumer::ImageWriteConsumer(const std::string &outputDir)
    : outputDir_(outputDir)
{
    if (outputDir_.empty())
    {
        throw std::runtime_error("Output directory is empty");
    }

    fs::create_directories(outputDir_);
    LOG.info("ImageWriteConsumer output=%s", outputDir_.string().c_str());
}

void ImageWriteConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet)
{
    if (!packet)
    {
        return;
    }

    auto &p = static_cast<RealesrganPacket &>(*packet);
    if (p.outputBgrU8.empty())
    {
        LOG.warning("Output image is empty for idx=%d", p.idx);
        return;
    }

    std::string fileName = p.filename;
    if (fileName.empty())
    {
        fileName = "sr_output_" + std::to_string(p.idx) + ".png";
    }

    const auto outPath = outputDir_ / fileName;
    if (!cv::imwrite(outPath.string(), p.outputBgrU8))
    {
        LOG.error("Failed to write output image: %s", outPath.string().c_str());
        return;
    }

    ++writtenCount_;
}
