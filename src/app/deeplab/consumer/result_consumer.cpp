#include "consumer/result_consumer.h"

#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <iomanip>
#include <sstream>
#include <stdexcept>

DeeplabResultConsumer::DeeplabResultConsumer(const std::string &outputDir)
    : outputDir_(outputDir)
{
    if (outputDir_.empty())
    {
        throw std::runtime_error("Output directory is empty");
    }

    fs::create_directories(outputDir_);
    overlayDir_ = outputDir_ / "overlay";
    maskDir_ = outputDir_ / "mask";
    fs::create_directories(overlayDir_);
    fs::create_directories(maskDir_);

    LOG.info("DeeplabResultConsumer output root=%s", outputDir_.string().c_str());
    LOG.info("DeeplabResultConsumer overlay dir=%s", overlayDir_.string().c_str());
    LOG.info("DeeplabResultConsumer mask dir=%s", maskDir_.string().c_str());
}

void DeeplabResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet)
{
    if (!packet)
    {
        return;
    }

    consumedCount_.fetch_add(1, std::memory_order_relaxed);

    auto &p = static_cast<DeeplabPacket &>(*packet);
    if (p.originalImage.empty() || p.mask.empty())
    {
        LOG.warning("Frame idx=%d has empty original image or mask", p.idx);
        return;
    }

    // VOC palette in BGR order (OpenCV write/read convention).
    static const cv::Vec3b kVocColorsBgr[21] = {
        {0, 0, 0},       {0, 0, 128},    {0, 128, 0},    {0, 128, 128}, {128, 0, 0},
        {128, 0, 128},   {128, 128, 0},  {128, 128, 128}, {0, 0, 64},    {0, 0, 192},
        {0, 128, 64},    {0, 128, 192},  {128, 0, 64},    {128, 0, 192}, {128, 128, 64},
        {128, 128, 192}, {0, 64, 0},     {0, 64, 128},    {0, 192, 0},   {0, 192, 128},
        {128, 64, 0}};

    double minClass = 0.0;
    double maxClass = 0.0;
    cv::minMaxLoc(p.mask, &minClass, &maxClass);
    const double nonBackgroundRatio =
        static_cast<double>(cv::countNonZero(p.mask)) / static_cast<double>(p.mask.total());
    LOG.info(
        "Frame idx=%d mask class range=[%.0f, %.0f], non-background ratio=%.4f",
        p.idx,
        minClass,
        maxClass,
        nonBackgroundRatio);
    if (nonBackgroundRatio < 0.001)
    {
        LOG.warning("Frame idx=%d prediction is near-all-background", p.idx);
    }

    cv::Mat colorMask(p.mask.size(), CV_8UC3);
    int outOfRangeCount = 0;
    for (int y = 0; y < p.mask.rows; ++y)
    {
        const auto *maskRow = p.mask.ptr<uchar>(y);
        auto *dstRow = colorMask.ptr<cv::Vec3b>(y);
        for (int x = 0; x < p.mask.cols; ++x)
        {
            const int cls = static_cast<int>(maskRow[x]);
            if (cls >= 0 && cls < 21)
            {
                dstRow[x] = kVocColorsBgr[cls];
            }
            else
            {
                dstRow[x] = kVocColorsBgr[0];
                ++outOfRangeCount;
            }
        }
    }
    if (outOfRangeCount > 0)
    {
        LOG.warning("Frame idx=%d has %d out-of-range class pixels", p.idx, outOfRangeCount);
    }

    cv::Mat blended;
    cv::addWeighted(p.originalImage, 0.6, colorMask, 0.4, 0.0, blended);

    std::ostringstream baseName;
    baseName << "deeplab_" << std::setfill('0') << std::setw(6) << p.idx;

    const auto overlayPath = overlayDir_ / (baseName.str() + ".jpg");
    const auto maskPath = maskDir_ / (baseName.str() + ".png");

    if (!cv::imwrite(overlayPath.string(), blended))
    {
        LOG.error("Failed to write overlay image: %s", overlayPath.string().c_str());
        return;
    }

    if (!cv::imwrite(maskPath.string(), colorMask))
    {
        LOG.error("Failed to write mask image: %s", maskPath.string().c_str());
        return;
    }

    writtenCount_.fetch_add(1, std::memory_order_relaxed);
}
