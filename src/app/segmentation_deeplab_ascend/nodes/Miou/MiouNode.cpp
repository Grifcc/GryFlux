#include "MiouNode.h"

#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

namespace PipelineNodes
{

void MiouNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeepLabPacket &>(packet);

    if (p.pred_mask_resized.empty())
    {
        throw std::runtime_error("MiouNode: pred_mask_resized is empty.");
    }
    if (p.gt_mask.empty())
    {
        throw std::runtime_error("MiouNode: gt_mask is empty.");
    }
    if (p.pred_mask_resized.size() != p.gt_mask.size())
    {
        throw std::runtime_error("MiouNode: prediction and gt sizes do not match.");
    }

    std::vector<uint64_t> intersections(NUM_CLASSES, 0);
    std::vector<uint64_t> unions(NUM_CLASSES, 0);

    for (int y = 0; y < p.gt_mask.rows; ++y)
    {
        const auto *predRow = p.pred_mask_resized.ptr<unsigned char>(y);
        const auto *gtRow = p.gt_mask.ptr<unsigned char>(y);

        for (int x = 0; x < p.gt_mask.cols; ++x)
        {
            const int pred = static_cast<int>(predRow[x]);
            const int gt = static_cast<int>(gtRow[x]);

            if (gt < 0 || gt >= NUM_CLASSES)
            {
                continue;
            }
            if (pred < 0 || pred >= NUM_CLASSES)
            {
                continue;
            }

            if (pred == gt)
            {
                intersections[gt]++;
            }

            unions[gt]++;
            if (pred != gt)
            {
                unions[pred]++;
            }
        }
    }

    double miou = 0.0;
    int validClasses = 0;
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        if (unions[c] == 0)
        {
            continue;
        }

        miou += static_cast<double>(intersections[c]) / static_cast<double>(unions[c]);
        ++validClasses;
    }

    p.miou = (validClasses > 0)
        ? static_cast<float>(miou / static_cast<double>(validClasses))
        : 0.0f;

    LOG.debug("Packet %d: miou = %.6f", p.frame_id, p.miou);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
