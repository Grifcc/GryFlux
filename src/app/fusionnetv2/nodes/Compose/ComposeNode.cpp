#include "nodes/Compose/ComposeNode.h"

#include "packet/fusionnetv2_packet.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>
#include <vector>

namespace FusionNetV2Nodes
{

void ComposeNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<FusionNetV2Packet &>(packet);

    if (p.fusedYF32.empty())
    {
        throw std::runtime_error("ComposeNode received empty fused Y output");
    }
    if (p.visCbU8.empty() || p.visCrU8.empty())
    {
        throw std::runtime_error("ComposeNode missing visible chroma channels");
    }

    cv::Mat fusedYScaled = p.fusedYF32.clone();

    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(fusedYScaled, &minValue, &maxValue);

    if (maxValue <= 1.5)
    {
        fusedYScaled *= 255.0f;
    }

    cv::max(fusedYScaled, 0.0f, fusedYScaled);
    cv::min(fusedYScaled, 255.0f, fusedYScaled);

    cv::Mat fusedYU8;
    fusedYScaled.convertTo(fusedYU8, CV_8UC1);

    std::vector<cv::Mat> yCrCb = {
        fusedYU8,
        p.visCrU8,
        p.visCbU8,
    };

    cv::Mat merged;
    cv::merge(yCrCb, merged);
    cv::cvtColor(merged, p.outputBgrU8, cv::COLOR_YCrCb2BGR);
}

} // namespace FusionNetV2Nodes
