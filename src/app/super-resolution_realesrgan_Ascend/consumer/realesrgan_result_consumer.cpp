#include "realesrgan_result_consumer.h"

#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include <stdexcept>

RealEsrganResultConsumer::RealEsrganResultConsumer(size_t expectedTotal)
    : expectedTotal_(expectedTotal)
{
}

void RealEsrganResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet)
{
    auto &p = static_cast<RealEsrganPacket &>(*packet);
    if (p.sr_image.empty())
    {
        throw std::runtime_error("RealEsrganResultConsumer: sr_image is empty.");
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

size_t RealEsrganResultConsumer::getConsumedCount() const
{
    return consumedCount_.load(std::memory_order_relaxed);
}

void RealEsrganResultConsumer::printSummary() const
{
    const size_t consumed = getConsumedCount();
    LOG.info("========================================");
    LOG.info("RealESRGAN Result Summary");
    LOG.info("Consumed packets: %zu", consumed);
    LOG.info("========================================");
}
