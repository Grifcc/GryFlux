#include "InputNode.h"

#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace PipelineNodes
{

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeepLabPacket &>(packet);

    if (p.image_path.empty())
    {
        throw std::runtime_error("InputNode: image_path is empty.");
    }

    const std::filesystem::path imagePath(p.image_path);
    if (!std::filesystem::exists(imagePath))
    {
        throw std::runtime_error("InputNode: image file does not exist: " + p.image_path);
    }

    p.orig_w = 0;
    p.orig_h = 0;
    p.pred_mask_resized.release();

    LOG.debug(
        "Packet %d: image=%s",
        p.frame_id,
        p.image_path.c_str());

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
