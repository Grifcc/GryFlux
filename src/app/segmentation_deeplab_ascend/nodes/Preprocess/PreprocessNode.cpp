#include "PreprocessNode.h"

#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace PipelineNodes
{

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeepLabPacket &>(packet);

    cv::Mat image = cv::imread(p.image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        throw std::runtime_error("PreprocessNode: failed to read image: " + p.image_path);
    }

    p.orig_w = image.cols;
    p.orig_h = image.rows;

    cv::Mat resized;
    cv::resize(
        image,
        resized,
        cv::Size(MODEL_INPUT_W, MODEL_INPUT_H),
        0.0,
        0.0,
        cv::INTER_LINEAR);

    cv::Mat rgbFloat;
    resized.convertTo(rgbFloat, CV_32FC3, 1.0 / 127.5, -1.0);
    cv::cvtColor(rgbFloat, rgbFloat, cv::COLOR_BGR2RGB);

    if (p.input_tensor.size() != static_cast<size_t>(3 * MODEL_INPUT_H * MODEL_INPUT_W))
    {
        p.input_tensor.resize(3 * MODEL_INPUT_H * MODEL_INPUT_W);
    }

    const size_t planeSize = static_cast<size_t>(MODEL_INPUT_H * MODEL_INPUT_W);
    std::vector<cv::Mat> channels(3);
    cv::split(rgbFloat, channels);
    for (int c = 0; c < 3; ++c)
    {
        std::memcpy(
            p.input_tensor.data() + static_cast<size_t>(c) * planeSize,
            channels[c].ptr<float>(),
            planeSize * sizeof(float));
    }

    LOG.debug(
        "Packet %d: preprocess done, original size=%dx%d",
        p.frame_id,
        p.orig_w,
        p.orig_h);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
