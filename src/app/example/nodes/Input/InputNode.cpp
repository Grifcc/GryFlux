#include "InputNode.h"

#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>

namespace PipelineNodes
{

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    const float x = static_cast<float>(p.id);
    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.aVec[i] = x;
    }

    LOG.debug("Packet %d: a[0] = %.1f", p.id, p.aVec[0]);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
