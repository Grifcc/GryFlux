#pragma once

#include "framework/node_base.h"

#include <filesystem>
#include <mutex>
#include <string>

namespace cv
{
class Mat;
}

namespace PipelineNodes
{

class OutputNode : public GryFlux::NodeBase
{
public:
    explicit OutputNode(std::string outputDir = "deeplab_outputs",
                        size_t maxSavedImages = 10);

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    std::string saveOverlayImage(const cv::Mat &overlayImage,
                                 const std::string &imagePath,
                                 int frameId);

    std::filesystem::path buildOutputPath(const std::string &imagePath, int frameId) const;

    std::filesystem::path outputDir_;
    size_t maxSavedImages_;
    std::mutex mutex_;
    size_t savedCount_ = 0;
};

} // namespace PipelineNodes
