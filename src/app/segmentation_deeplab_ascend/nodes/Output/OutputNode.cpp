#include "OutputNode.h"

#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <stdexcept>

namespace PipelineNodes
{

namespace
{

constexpr double kOverlayAlpha = 0.65;
constexpr double kImageAlpha = 1.0 - kOverlayAlpha;

const std::array<cv::Vec3b, NUM_CLASSES> &GetClassPalette()
{
    static const std::array<cv::Vec3b, NUM_CLASSES> kPalette = {{
        cv::Vec3b(0, 0, 0),
        cv::Vec3b(0, 0, 192),
        cv::Vec3b(0, 192, 0),
        cv::Vec3b(0, 192, 192),
        cv::Vec3b(192, 0, 0),
        cv::Vec3b(192, 0, 192),
        cv::Vec3b(192, 192, 0),
        cv::Vec3b(192, 192, 192),
        cv::Vec3b(0, 0, 96),
        cv::Vec3b(0, 96, 0),
        cv::Vec3b(0, 96, 96),
        cv::Vec3b(96, 0, 0),
        cv::Vec3b(96, 0, 96),
        cv::Vec3b(96, 96, 0),
        cv::Vec3b(96, 96, 96),
        cv::Vec3b(255, 64, 0),
        cv::Vec3b(0, 255, 64),
        cv::Vec3b(64, 255, 255),
        cv::Vec3b(255, 64, 255),
        cv::Vec3b(255, 255, 64),
        cv::Vec3b(255, 255, 255),
    }};
    return kPalette;
}

cv::Mat NormalizeLabelMask(const cv::Mat &mask, const cv::Size &imageSize)
{
    cv::Mat labelMask;
    if (mask.type() == CV_8UC1)
    {
        labelMask = mask;
    }
    else
    {
        mask.convertTo(labelMask, CV_8UC1);
    }

    if (labelMask.size() != imageSize)
    {
        cv::resize(labelMask, labelMask, imageSize, 0.0, 0.0, cv::INTER_NEAREST);
    }

    return labelMask;
}

cv::Mat BuildOverlayImage(const cv::Mat &originalImage, const cv::Mat &labelMask)
{
    cv::Mat overlay = originalImage.clone();
    cv::Mat boundaryMask = cv::Mat::zeros(labelMask.size(), CV_8UC1);
    const auto &palette = GetClassPalette();

    for (int y = 0; y < labelMask.rows; ++y)
    {
        const auto *labelRow = labelMask.ptr<unsigned char>(y);
        const auto *prevRow = y > 0 ? labelMask.ptr<unsigned char>(y - 1) : nullptr;
        const auto *nextRow = (y + 1) < labelMask.rows ? labelMask.ptr<unsigned char>(y + 1) : nullptr;
        const auto *srcRow = originalImage.ptr<cv::Vec3b>(y);
        auto *dstRow = overlay.ptr<cv::Vec3b>(y);
        auto *boundaryRow = boundaryMask.ptr<unsigned char>(y);

        for (int x = 0; x < labelMask.cols; ++x)
        {
            const unsigned char cls = labelRow[x];
            if (cls == 0)
            {
                continue;
            }

            const cv::Vec3b &src = srcRow[x];
            const cv::Vec3b &color = palette[cls];
            dstRow[x] = cv::Vec3b(
                cv::saturate_cast<unsigned char>(src[0] * kImageAlpha + color[0] * kOverlayAlpha),
                cv::saturate_cast<unsigned char>(src[1] * kImageAlpha + color[1] * kOverlayAlpha),
                cv::saturate_cast<unsigned char>(src[2] * kImageAlpha + color[2] * kOverlayAlpha));

            const bool leftDifferent = x > 0 && cls != labelRow[x - 1];
            const bool rightDifferent = (x + 1) < labelMask.cols && cls != labelRow[x + 1];
            const bool upDifferent = prevRow != nullptr && cls != prevRow[x];
            const bool downDifferent = nextRow != nullptr && cls != nextRow[x];
            if (leftDifferent || rightDifferent || upDifferent || downDifferent)
            {
                boundaryRow[x] = 255;
            }
        }
    }

    cv::dilate(boundaryMask, boundaryMask, cv::Mat(), cv::Point(-1, -1), 1);
    overlay.setTo(cv::Scalar(255, 255, 255), boundaryMask);
    return overlay;
}

} // namespace

OutputNode::OutputNode(std::string outputDir, size_t maxSavedImages)
    : outputDir_(std::move(outputDir)),
      maxSavedImages_(maxSavedImages)
{
    std::filesystem::create_directories(outputDir_);
}

void OutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeepLabPacket &>(packet);

    if (p.pred_mask_resized.empty())
    {
        throw std::runtime_error("OutputNode: pred_mask_resized is empty.");
    }

    cv::Mat originalImage = cv::imread(p.image_path, cv::IMREAD_COLOR);
    if (originalImage.empty())
    {
        throw std::runtime_error("OutputNode: failed to read image: " + p.image_path);
    }

    const cv::Mat labelMask = NormalizeLabelMask(p.pred_mask_resized, originalImage.size());
    const cv::Mat overlayImage = BuildOverlayImage(originalImage, labelMask);
    const std::string savedPath = saveOverlayImage(overlayImage, p.image_path, p.frame_id);

    if (!savedPath.empty())
    {
        LOG.info("Packet %d: segmentation done, saved=%s", p.frame_id, savedPath.c_str());
    }
    else
    {
        LOG.info("Packet %d: segmentation done", p.frame_id);
    }
}

std::string OutputNode::saveOverlayImage(const cv::Mat &overlayImage,
                                         const std::string &imagePath,
                                         int frameId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (savedCount_ >= maxSavedImages_)
    {
        return {};
    }

    const auto outputPath = buildOutputPath(imagePath, frameId);
    if (!cv::imwrite(outputPath.string(), overlayImage))
    {
        throw std::runtime_error("OutputNode: failed to save image: " + outputPath.string());
    }

    ++savedCount_;
    return outputPath.string();
}

std::filesystem::path OutputNode::buildOutputPath(const std::string &imagePath, int frameId) const
{
    const std::filesystem::path inputPath(imagePath);
    const std::string stem = inputPath.stem().empty() ? "frame" : inputPath.stem().string();
    return outputDir_ / (std::to_string(frameId) + "_" + stem + "_overlay.png");
}

} // namespace PipelineNodes
