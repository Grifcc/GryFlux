#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/realesrgan_packet.h"

#include <opencv2/opencv.hpp>

#include <stdexcept>

namespace RealesrganNodes
{

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<RealesrganPacket &>(packet);

    if (p.srTensorF32.empty())
    {
        throw std::runtime_error("Postprocess: empty SR tensor");
    }
    if (p.srTensorF32.channels() != 3)
    {
        throw std::runtime_error("Postprocess: expected 3-channel SR tensor");
    }

    cv::Mat srFloat;
    if (p.srTensorF32.type() != CV_32FC3)
    {
        p.srTensorF32.convertTo(srFloat, CV_32FC3);
    }
    else
    {
        srFloat = p.srTensorF32;
    }

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(srFloat, &minVal, &maxVal);

    cv::Mat scaled;
    if (maxVal <= 2.0)
    {
        srFloat.convertTo(scaled, CV_32FC3, 255.0);
    }
    else
    {
        scaled = srFloat;
    }

    cv::max(scaled, 0.0, scaled);
    cv::min(scaled, 255.0, scaled);

    cv::Mat srU8;
    scaled.convertTo(srU8, CV_8UC3);

    if (srU8.channels() != 3)
    {
        throw std::runtime_error("Postprocess: expected 3-channel output");
    }

    cv::cvtColor(srU8, p.outputBgrU8, cv::COLOR_RGB2BGR);
}

} // namespace RealesrganNodes
