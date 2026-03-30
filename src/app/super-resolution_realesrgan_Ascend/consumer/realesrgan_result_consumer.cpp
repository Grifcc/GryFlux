#include "realesrgan_result_consumer.h"

#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include <cmath>
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

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (p.has_valid_psnr && std::isfinite(p.psnr))
        {
            totalPsnr_ += p.psnr;
            ++validPsnrCount_;
        }
    }

    const size_t consumed = consumedCount_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (expectedTotal_ > 0)
    {
        if (p.has_valid_psnr)
        {
            LOG.info("Packet %d: super-resolution done (%zu/%zu), output=%dx%d, PSNR=%.4f dB",
                     p.frame_id,
                     consumed,
                     expectedTotal_,
                     p.sr_image.cols,
                     p.sr_image.rows,
                     p.psnr);
        }
        else
        {
            LOG.info("Packet %d: super-resolution done (%zu/%zu), output=%dx%d, PSNR=N/A",
                     p.frame_id,
                     consumed,
                     expectedTotal_,
                     p.sr_image.cols,
                     p.sr_image.rows);
        }
    }
    else if (p.has_valid_psnr)
    {
        LOG.info("Packet %d: super-resolution done, output=%dx%d, PSNR=%.4f dB",
                 p.frame_id,
                 p.sr_image.cols,
                 p.sr_image.rows,
                 p.psnr);
    }
    else
    {
        LOG.info("Packet %d: super-resolution done, output=%dx%d, PSNR=N/A",
                 p.frame_id,
                 p.sr_image.cols,
                 p.sr_image.rows);
    }
}

size_t RealEsrganResultConsumer::getConsumedCount() const
{
    return consumedCount_.load(std::memory_order_relaxed);
}

void RealEsrganResultConsumer::printSummary() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t consumed = getConsumedCount();
    LOG.info("========================================");
    LOG.info("RealESRGAN Result Summary");
    LOG.info("Consumed packets: %zu", consumed);
    if (validPsnrCount_ > 0)
    {
        LOG.info("Valid PSNR packets: %zu", validPsnrCount_);
        LOG.info("Average PSNR: %.4f dB", totalPsnr_ / static_cast<double>(validPsnrCount_));
    }
    else
    {
        LOG.info("Valid PSNR packets: 0");
        LOG.info("Average PSNR: N/A");
    }
    LOG.info("========================================");
}
