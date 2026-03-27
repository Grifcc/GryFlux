#include "deeplab_result_consumer.h"

#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <opencv2/imgproc.hpp>

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace
{

constexpr int kIgnoreLabel = 255;

const std::vector<std::string> kVocClassNames = {
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
};

} // namespace

DeepLabResultConsumer::DeepLabResultConsumer(size_t expectedTotal)
    : expectedTotal_(expectedTotal),
      classNames_(kVocClassNames),
      hist_(NUM_CLASSES * NUM_CLASSES, 0)
{
}

void DeepLabResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet)
{
    auto &p = static_cast<DeepLabPacket &>(*packet);

    if (p.pred_mask_resized.empty())
    {
        throw std::runtime_error("DeepLabResultConsumer: pred_mask_resized is empty.");
    }
    if (p.gt_mask.empty())
    {
        throw std::runtime_error("DeepLabResultConsumer: gt_mask is empty.");
    }

    updateConfusionMatrix(p.pred_mask_resized, p.gt_mask);

    const size_t consumed = consumedCount_.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        totalPacketMiou_ += static_cast<double>(p.miou);
    }

    if (expectedTotal_ > 0)
    {
        LOG.info(
            "Packet %d: image MIoU = %.4f (%zu/%zu)",
            p.frame_id,
            p.miou,
            consumed,
            expectedTotal_);
    }
    else
    {
        LOG.info("Packet %d: image MIoU = %.4f", p.frame_id, p.miou);
    }
}

size_t DeepLabResultConsumer::getConsumedCount() const
{
    return consumedCount_.load(std::memory_order_relaxed);
}

void DeepLabResultConsumer::printSummary() const
{
    std::vector<int64_t> histCopy;
    double totalPacketMiou = 0.0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        histCopy = hist_;
        totalPacketMiou = totalPacketMiou_;
    }

    const size_t consumed = getConsumedCount();
    LOG.info("========================================");
    LOG.info("DeepLab Result Summary");
    LOG.info("Consumed packets: %zu", consumed);

    if (consumed == 0)
    {
        LOG.info("No packets were consumed.");
        LOG.info("========================================");
        return;
    }

    std::vector<int64_t> intersections(NUM_CLASSES, 0);
    std::vector<int64_t> predTotals(NUM_CLASSES, 0);
    std::vector<int64_t> gtTotals(NUM_CLASSES, 0);

    for (int pred = 0; pred < NUM_CLASSES; ++pred)
    {
        for (int gt = 0; gt < NUM_CLASSES; ++gt)
        {
            const int64_t count = histCopy[static_cast<size_t>(pred) * NUM_CLASSES + static_cast<size_t>(gt)];
            if (pred == gt)
            {
                intersections[pred] = count;
            }
            predTotals[pred] += count;
            gtTotals[gt] += count;
        }
    }

    double datasetMiou = 0.0;
    int validClasses = 0;
    for (int cls = 0; cls < NUM_CLASSES; ++cls)
    {
        if (gtTotals[cls] == 0)
        {
            continue;
        }

        const int64_t unionCount = predTotals[cls] + gtTotals[cls] - intersections[cls];
        const double iou = (unionCount > 0)
            ? static_cast<double>(intersections[cls]) / static_cast<double>(unionCount)
            : 0.0;

        const std::string className = (static_cast<size_t>(cls) < classNames_.size())
            ? classNames_[cls]
            : std::to_string(cls);

        LOG.info("Class %d (%s): IoU = %.4f%%", cls, className.c_str(), iou * 100.0);
        datasetMiou += iou;
        ++validClasses;
    }

    if (validClasses > 0)
    {
        datasetMiou /= static_cast<double>(validClasses);
    }

    const double avgPacketMiou = totalPacketMiou / static_cast<double>(consumed);
    LOG.info("Average packet MIoU: %.4f%%", avgPacketMiou * 100.0);
    LOG.info("Dataset MIoU: %.4f%%", datasetMiou * 100.0);
    LOG.info("========================================");
}

void DeepLabResultConsumer::updateConfusionMatrix(const cv::Mat &predMask, const cv::Mat &gtMask)
{
    cv::Mat gtAligned;
    if (predMask.size() != gtMask.size())
    {
        cv::resize(gtMask, gtAligned, predMask.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    else
    {
        gtAligned = gtMask;
    }

    if (predMask.type() != CV_8UC1 || gtAligned.type() != CV_8UC1)
    {
        throw std::runtime_error("DeepLabResultConsumer: masks must be CV_8UC1.");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (int row = 0; row < predMask.rows; ++row)
    {
        const auto *predPtr = predMask.ptr<unsigned char>(row);
        const auto *gtPtr = gtAligned.ptr<unsigned char>(row);

        for (int col = 0; col < predMask.cols; ++col)
        {
            const int pred = static_cast<int>(predPtr[col]);
            const int gt = static_cast<int>(gtPtr[col]);

            if (gt == kIgnoreLabel || gt < 0 || gt >= NUM_CLASSES)
            {
                continue;
            }
            if (pred < 0 || pred >= NUM_CLASSES)
            {
                continue;
            }

            hist_[static_cast<size_t>(pred) * NUM_CLASSES + static_cast<size_t>(gt)]++;
        }
    }
}
