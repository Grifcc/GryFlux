#include "deeplab_result_consumer.h"

#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <stdexcept>

DeepLabResultConsumer::DeepLabResultConsumer(size_t expectedTotal)
    : expectedTotal_(expectedTotal)
{
}

void DeepLabResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet)
{
    auto &p = static_cast<DeepLabPacket &>(*packet);

    if (p.pred_mask_resized.empty())
    {
        throw std::runtime_error("DeepLabResultConsumer: pred_mask_resized is empty.");
    }

    const size_t consumed = consumedCount_.fetch_add(1, std::memory_order_relaxed) + 1;

    if (expectedTotal_ > 0)
    {
        LOG.debug("Packet %d: consumed (%zu/%zu)", p.frame_id, consumed, expectedTotal_);
    }
    else
    {
        LOG.debug("Packet %d: consumed", p.frame_id);
    }
}

size_t DeepLabResultConsumer::getConsumedCount() const
{
    return consumedCount_.load(std::memory_order_relaxed);
}

void DeepLabResultConsumer::printSummary() const
{
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
    LOG.info("========================================");
}
