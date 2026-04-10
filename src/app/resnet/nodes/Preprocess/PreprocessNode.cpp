#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/resnet_packet.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <stdexcept>

namespace ResnetNodes
{

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<ResnetPacket &>(packet);

    if (p.originalImage.empty())
    {
        throw std::runtime_error("Preprocess: empty input image");
    }

    cv::Mat rgb;
    cv::cvtColor(p.originalImage, rgb, cv::COLOR_BGR2RGB);

    const int imageWidth = p.originalImage.cols;
    const int imageHeight = p.originalImage.rows;
    const float scale = std::min(static_cast<float>(modelWidth_) / static_cast<float>(imageWidth),
                                 static_cast<float>(modelHeight_) / static_cast<float>(imageHeight));

    const int resizedWidth = std::max(1, static_cast<int>(imageWidth * scale));
    const int resizedHeight = std::max(1, static_cast<int>(imageHeight * scale));

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resizedWidth, resizedHeight));

    cv::Mat letterbox(static_cast<int>(modelHeight_), static_cast<int>(modelWidth_), CV_8UC3, cv::Scalar(114, 114, 114));
    const int xPad = (static_cast<int>(modelWidth_) - resizedWidth) / 2;
    const int yPad = (static_cast<int>(modelHeight_) - resizedHeight) / 2;
    resized.copyTo(letterbox(cv::Rect(xPad, yPad, resizedWidth, resizedHeight)));

    p.preprocessedImage.create(static_cast<int>(modelHeight_), static_cast<int>(modelWidth_), CV_8UC3);
    letterbox.copyTo(p.preprocessedImage);
    p.modelWidth = modelWidth_;
    p.modelHeight = modelHeight_;

    const std::size_t inputBytes = modelWidth_ * modelHeight_ * 3;
    if (p.inputData.size() != inputBytes)
    {
        p.inputData.resize(inputBytes);
    }

    std::size_t offset = 0;
    for (int h = 0; h < static_cast<int>(modelHeight_); ++h)
    {
        const auto *row = p.preprocessedImage.ptr<uint8_t>(h);
        for (int w = 0; w < static_cast<int>(modelWidth_); ++w)
        {
            p.inputData[offset++] = row[w * 3 + 0];
            p.inputData[offset++] = row[w * 3 + 1];
            p.inputData[offset++] = row[w * 3 + 2];
        }
    }
}

} // namespace ResnetNodes

