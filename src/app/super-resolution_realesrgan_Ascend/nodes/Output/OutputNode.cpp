#include "OutputNode.h"

#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include <opencv2/imgcodecs.hpp>

#include <sstream>
#include <stdexcept>

namespace PipelineNodes
{

OutputNode::OutputNode(std::string outputDir, size_t maxSavedImages)
    : outputDir_(std::move(outputDir)),
      maxSavedImages_(maxSavedImages)
{
    std::filesystem::create_directories(outputDir_);
}

void OutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<RealEsrganPacket &>(packet);

    if (p.sr_image.empty())
    {
        throw std::runtime_error("OutputNode: sr_image is empty.");
    }

    std::string savedPath;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (savedCount_ < maxSavedImages_)
        {
            const auto outputPath = buildOutputPath(p.lr_path, p.frame_id);
            if (!cv::imwrite(outputPath.string(), p.sr_image))
            {
                throw std::runtime_error("OutputNode: failed to save image: " + outputPath.string());
            }
            ++savedCount_;
            savedPath = outputPath.string();
        }
    }

    if (!savedPath.empty())
    {
        LOG.info("Packet %d: super-resolution done, saved=%s", p.frame_id, savedPath.c_str());
    }
    else
    {
        LOG.info("Packet %d: super-resolution done", p.frame_id);
    }
}

std::filesystem::path OutputNode::buildOutputPath(const std::string &imagePath, int frameId) const
{
    std::ostringstream oss;
    const std::filesystem::path inputPath(imagePath);
    const std::string stem = inputPath.stem().empty() ? "frame" : inputPath.stem().string();
    oss << frameId << "_" << stem << "_sr.png";
    return outputDir_ / oss.str();
}

} // namespace PipelineNodes
