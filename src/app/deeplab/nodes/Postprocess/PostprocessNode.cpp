#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/deeplab_packet.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <stdexcept>

namespace DeeplabNodes
{

namespace
{
constexpr int kClassCount = 21;

int clampInt(int value, int minValue, int maxValue)
{
    return std::max(minValue, std::min(value, maxValue));
}
} // namespace

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeeplabPacket &>(packet);

    if (p.inferenceOutputs.empty())
    {
        throw std::runtime_error("Postprocess: no inference outputs");
    }

    const auto &output = p.inferenceOutputs.front();
    if (output.data.empty() || output.gridH == 0 || output.gridW == 0)
    {
        throw std::runtime_error("Postprocess: invalid Deeplab output tensor");
    }
    if (output.channels != kClassCount)
    {
        throw std::runtime_error("Postprocess: unexpected Deeplab class count");
    }

    cv::Mat logits(static_cast<int>(output.gridH),
                   static_cast<int>(output.gridW),
                   CV_32FC(kClassCount),
                   const_cast<float*>(output.data.data()));

    cv::Mat resizedLogits;
    cv::resize(logits,
               resizedLogits,
               cv::Size(static_cast<int>(p.modelWidth), static_cast<int>(p.modelHeight)),
               0,
               0,
               cv::INTER_LINEAR);

    const int cropX = clampInt(p.xPad, 0, resizedLogits.cols - 1);
    const int cropY = clampInt(p.yPad, 0, resizedLogits.rows - 1);
    const int expectedCropW = static_cast<int>(p.resizedWidth);
    const int expectedCropH = static_cast<int>(p.resizedHeight);
    const int cropW = clampInt(expectedCropW, 1, resizedLogits.cols - cropX);
    const int cropH = clampInt(expectedCropH, 1, resizedLogits.rows - cropY);

    const cv::Rect roi(cropX, cropY, cropW, cropH);
    const cv::Mat croppedLogits = resizedLogits(roi);
    
    cv::Mat originalSizeLogits;
    cv::resize(croppedLogits, originalSizeLogits, p.originalImage.size(), 0, 0, cv::INTER_LINEAR);

    p.mask.create(p.originalImage.rows, p.originalImage.cols, CV_8UC1);
    for (int y = 0; y < p.originalImage.rows; ++y)
    {
        const float* rowPtr = originalSizeLogits.ptr<float>(y);
        uchar* maskPtr = p.mask.ptr<uchar>(y);
        for (int x = 0; x < p.originalImage.cols; ++x)
        {
            int bestClass = 0;
            float bestScore = rowPtr[x * kClassCount];
            for (int c = 1; c < kClassCount; ++c)
            {
                float score = rowPtr[x * kClassCount + c];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }
            maskPtr[x] = static_cast<uchar>(bestClass);
        }
    }
}

} // namespace DeeplabNodes
