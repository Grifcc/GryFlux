#pragma once

#include "framework/node_base.h"

#include <filesystem>
#include <limits>
#include <mutex>
#include <string>

namespace PipelineNodes
{

class OutputNode : public GryFlux::NodeBase
{
public:
    explicit OutputNode(
        std::string outputDir = "realesrgan_outputs",
        size_t maxSavedImages = std::numeric_limits<size_t>::max());

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    std::filesystem::path buildOutputPath(const std::string &imagePath, int frameId) const;

    std::filesystem::path outputDir_;
    size_t maxSavedImages_;
    std::mutex mutex_;
    size_t savedCount_ = 0;
};

} // namespace PipelineNodes
