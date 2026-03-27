#include "GtProcessNode.h"

#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <array>
#include <stdexcept>
#include <thread>

namespace PipelineNodes
{

namespace
{

constexpr unsigned char kIgnoreLabel = 255;

std::array<unsigned char, 1 << 24> buildVocColorLut()
{
    std::array<unsigned char, 1 << 24> lut{};
    lut.fill(kIgnoreLabel);

    const std::array<cv::Vec3b, NUM_CLASSES> vocColorMap = {{
        {0, 0, 0},
        {0, 0, 128},
        {0, 128, 0},
        {0, 128, 128},
        {128, 0, 0},
        {128, 0, 128},
        {128, 128, 0},
        {128, 128, 128},
        {0, 0, 64},
        {0, 0, 192},
        {0, 128, 64},
        {0, 128, 192},
        {128, 0, 64},
        {128, 0, 192},
        {128, 128, 64},
        {128, 128, 192},
        {0, 64, 0},
        {0, 64, 128},
        {0, 192, 0},
        {0, 192, 128},
        {128, 64, 0},
    }};

    for (size_t index = 0; index < vocColorMap.size(); ++index)
    {
        const auto &color = vocColorMap[index];
        const int key = (static_cast<int>(color[0]) << 16)
            | (static_cast<int>(color[1]) << 8)
            | static_cast<int>(color[2]);
        lut[static_cast<size_t>(key)] = static_cast<unsigned char>(index);
    }

    return lut;
}

const std::array<unsigned char, 1 << 24> &vocColorLut()
{
    static const std::array<unsigned char, 1 << 24> lut = buildVocColorLut();
    return lut;
}

} // namespace

void GtProcessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeepLabPacket &>(packet);

    if (p.gt_path.empty())
    {
        throw std::runtime_error("GtProcessNode: gt_path is empty.");
    }

    cv::Mat gt = cv::imread(p.gt_path, cv::IMREAD_COLOR);
    if (gt.empty())
    {
        throw std::runtime_error("GtProcessNode: failed to read gt image: " + p.gt_path);
    }

    if (gt.channels() != 3)
    {
        throw std::runtime_error("GtProcessNode: expected 3-channel VOC color mask.");
    }

    p.gt_mask.create(gt.rows, gt.cols, CV_8UC1);
    const auto &lut = vocColorLut();
    for (int row = 0; row < gt.rows; ++row)
    {
        const auto *src = gt.ptr<cv::Vec3b>(row);
        auto *dst = p.gt_mask.ptr<unsigned char>(row);
        for (int col = 0; col < gt.cols; ++col)
        {
            const auto &color = src[col];
            const int key = (static_cast<int>(color[0]) << 16)
                | (static_cast<int>(color[1]) << 8)
                | static_cast<int>(color[2]);
            dst[col] = lut[static_cast<size_t>(key)];
        }
    }

    LOG.debug(
        "Packet %d: gt process done, gt size=%dx%d",
        p.frame_id,
        p.gt_mask.cols,
        p.gt_mask.rows);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
