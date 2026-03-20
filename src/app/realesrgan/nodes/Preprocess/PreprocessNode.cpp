#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/realesrgan_packet.h"

#include <opencv2/opencv.hpp>

#include <stdexcept>

namespace RealesrganNodes
{

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<RealesrganPacket &>(packet);

    if (p.inputBgrU8.empty())
    {
        throw std::runtime_error("Preprocess: empty input image");
    }

    cv::Mat bgr;
    switch (p.inputBgrU8.channels())
    {
    case 1:
        cv::cvtColor(p.inputBgrU8, bgr, cv::COLOR_GRAY2BGR);
        break;
    case 3:
        bgr = p.inputBgrU8;
        break;
    case 4:
        cv::cvtColor(p.inputBgrU8, bgr, cv::COLOR_BGRA2BGR);
        break;
    default:
        throw std::runtime_error("Preprocess: unsupported channel count");
    }

    if (bgr.depth() != CV_8U)
    {
        cv::Mat bgrU8;
        bgr.convertTo(bgrU8, CV_8U);
        bgr = bgrU8;
    }

    if (bgr.cols != modelWidth_ || bgr.rows != modelHeight_)
    {
        throw std::runtime_error("Preprocess: input size mismatch, expected fixed model size");
    }

    cv::cvtColor(bgr, p.modelRgbU8, cv::COLOR_BGR2RGB);
}

} // namespace RealesrganNodes
