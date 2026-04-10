#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/deeplab_packet.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <stdexcept>

namespace DeeplabNodes
{

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeeplabPacket &>(packet);

    if (p.originalImage.empty())
    {
        throw std::runtime_error("Preprocess: empty input image");
    }

    cv::Mat rgb;
    cv::cvtColor(p.originalImage, rgb, cv::COLOR_BGR2RGB);

    const int imageWidth = p.originalImage.cols;
    const int imageHeight = p.originalImage.rows;
    p.scale = std::min(static_cast<float>(modelWidth_) / static_cast<float>(imageWidth),
                       static_cast<float>(modelHeight_) / static_cast<float>(imageHeight));

    const int resizedWidth = std::max(1, static_cast<int>(imageWidth * p.scale));
    const int resizedHeight = std::max(1, static_cast<int>(imageHeight * p.scale));

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resizedWidth, resizedHeight));

    cv::Mat letterbox(static_cast<int>(modelHeight_), static_cast<int>(modelWidth_), CV_8UC3, cv::Scalar(114, 114, 114));
    p.xPad = (static_cast<int>(modelWidth_) - resizedWidth) / 2;
    p.yPad = (static_cast<int>(modelHeight_) - resizedHeight) / 2;
    p.resizedWidth = static_cast<std::size_t>(resizedWidth);
    p.resizedHeight = static_cast<std::size_t>(resizedHeight);
    resized.copyTo(letterbox(cv::Rect(p.xPad, p.yPad, resizedWidth, resizedHeight)));

    p.preprocessedImage = letterbox;
    p.modelWidth = modelWidth_;
    p.modelHeight = modelHeight_;

    p.inputData.resize(modelWidth_ * modelHeight_ * 3);
    std::size_t offset = 0;
    for (int h = 0; h < static_cast<int>(modelHeight_); ++h)
    {
        const auto *row = letterbox.ptr<uint8_t>(h);
        for (int w = 0; w < static_cast<int>(modelWidth_); ++w)
        {
            p.inputData[offset++] = row[w * 3 + 0];
            p.inputData[offset++] = row[w * 3 + 1];
            p.inputData[offset++] = row[w * 3 + 2];
        }
    }
}

} // namespace DeeplabNodes
