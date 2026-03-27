#include "PostprocessNode.h"

#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <thread>

namespace PipelineNodes
{

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeepLabPacket &>(packet);

    if (p.orig_w <= 0 || p.orig_h <= 0)
    {
        throw std::runtime_error("PostprocessNode: invalid original image size.");
    }

    if (p.output_tensor.size() != static_cast<size_t>(MODEL_OUT_H * MODEL_OUT_W * NUM_CLASSES))
    {
        throw std::runtime_error("PostprocessNode: output_tensor size is invalid.");
    }

    cv::Mat predMask(MODEL_OUT_H, MODEL_OUT_W, CV_8UC1);
    for (int h = 0; h < MODEL_OUT_H; ++h)
    {
        for (int w = 0; w < MODEL_OUT_W; ++w)
        {
            const size_t pixelBase = static_cast<size_t>(h * MODEL_OUT_W + w) * NUM_CLASSES;
            float bestScore = std::numeric_limits<float>::lowest();
            int bestClass = 0;

            for (int c = 0; c < NUM_CLASSES; ++c)
            {
                const float score = p.output_tensor[pixelBase + static_cast<size_t>(c)];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            predMask.at<unsigned char>(h, w) = static_cast<unsigned char>(bestClass);
        }
    }

    cv::resize(
        predMask,
        p.pred_mask_resized,
        cv::Size(p.orig_w, p.orig_h),
        0.0,
        0.0,
        cv::INTER_NEAREST);

    LOG.debug(
        "Packet %d: postprocess done, pred size=%dx%d",
        p.frame_id,
        p.pred_mask_resized.cols,
        p.pred_mask_resized.rows);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
