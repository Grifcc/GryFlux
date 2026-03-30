#include "PreprocessNode.h"

#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace PipelineNodes
{

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<RealEsrganPacket &>(packet);

    p.lr_image = cv::imread(p.lr_path, cv::IMREAD_COLOR);
    if (p.lr_image.empty())
    {
        throw std::runtime_error("PreprocessNode: failed to read lr image: " + p.lr_path);
    }

    p.lr_w = p.lr_image.cols;
    p.lr_h = p.lr_image.rows;

    if (!p.hr_path.empty() && std::filesystem::exists(p.hr_path))
    {
        p.hr_image = cv::imread(p.hr_path, cv::IMREAD_COLOR);
    }

    cv::Mat resized;
    cv::resize(
        p.lr_image,
        resized,
        cv::Size(REALESRGAN_INPUT_W, REALESRGAN_INPUT_H),
        0.0,
        0.0,
        cv::INTER_CUBIC);

    cv::Mat rgbFloat;
    cv::cvtColor(resized, rgbFloat, cv::COLOR_BGR2RGB);
    rgbFloat.convertTo(rgbFloat, CV_32FC3, 1.0 / 255.0);

    const size_t planeSize = static_cast<size_t>(REALESRGAN_INPUT_W * REALESRGAN_INPUT_H);
    std::vector<cv::Mat> channels(3);
    cv::split(rgbFloat, channels);
    for (int c = 0; c < 3; ++c)
    {
        std::memcpy(
            p.input_tensor.data() + static_cast<size_t>(c) * planeSize,
            channels[c].ptr<float>(),
            planeSize * sizeof(float));
    }

    LOG.debug("Packet %d: preprocess done, lr size=%dx%d", p.frame_id, p.lr_w, p.lr_h);
    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
