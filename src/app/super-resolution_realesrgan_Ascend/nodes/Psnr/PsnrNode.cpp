#include "PsnrNode.h"

#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace
{

double calculatePsnr(const cv::Mat &lhs, const cv::Mat &rhs)
{
    cv::Mat rhsAligned = rhs;
    if (lhs.size() != rhs.size())
    {
        cv::resize(rhs, rhsAligned, lhs.size(), 0.0, 0.0, cv::INTER_CUBIC);
    }

    cv::Mat diff;
    cv::absdiff(lhs, rhsAligned, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    const cv::Scalar sum = cv::sum(diff);
    const double sse = sum[0] + sum[1] + sum[2];
    if (sse <= 1e-10)
    {
        return std::numeric_limits<double>::infinity();
    }

    const double mse = sse / static_cast<double>(lhs.channels() * lhs.total());
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

} // namespace

namespace PipelineNodes
{

void PsnrNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<RealEsrganPacket &>(packet);

    if (p.sr_image.empty())
    {
        throw std::runtime_error("PsnrNode: sr_image is empty.");
    }

    p.has_valid_psnr = false;
    p.psnr = std::numeric_limits<double>::quiet_NaN();

    if (p.hr_image.empty())
    {
        LOG.debug("Packet %d: skip PSNR because hr_image is empty", p.frame_id);
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
        return;
    }

    p.psnr = calculatePsnr(p.sr_image, p.hr_image);
    p.has_valid_psnr = true;

    LOG.debug("Packet %d: psnr = %.6f dB", p.frame_id, p.psnr);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
