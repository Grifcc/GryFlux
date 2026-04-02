#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/fusionnetv2_packet.h"

#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <vector>

namespace FusionNetV2Nodes
{

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<FusionNetV2Packet &>(packet);

    if (p.visibleBgrU8.empty() || p.infraredGrayU8.empty())
    {
        throw std::runtime_error("PreprocessNode received empty input image");
    }

    p.originalVisibleSize = p.visibleBgrU8.size();

    const cv::Size targetSize(modelWidth_, modelHeight_);

    cv::Mat visibleResized;
    if (p.visibleBgrU8.size() != targetSize)
    {
        cv::resize(p.visibleBgrU8, visibleResized, targetSize, 0.0, 0.0, cv::INTER_LINEAR);
    }
    else
    {
        visibleResized = p.visibleBgrU8;
    }

    cv::Mat infraredResized;
    if (p.infraredGrayU8.size() != targetSize)
    {
        cv::resize(p.infraredGrayU8, infraredResized, targetSize, 0.0, 0.0, cv::INTER_LINEAR);
    }
    else
    {
        infraredResized = p.infraredGrayU8;
    }

    cv::Mat yCrCb;
    cv::cvtColor(visibleResized, yCrCb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(yCrCb, channels);
    if (channels.size() != 3)
    {
        throw std::runtime_error("PreprocessNode expected 3 YCrCb channels");
    }

    channels[0].convertTo(p.visYF32, CV_32FC1, 1.0f / 255.0f);
    p.visCrU8 = channels[1];
    p.visCbU8 = channels[2];
    infraredResized.convertTo(p.infraredF32, CV_32FC1, 1.0f / 255.0f);
}

} // namespace FusionNetV2Nodes
