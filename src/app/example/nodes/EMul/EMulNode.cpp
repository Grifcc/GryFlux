#include "EMulNode.h"

#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>

namespace PipelineNodes
{

void EMulNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.eVec[i] = p.bVec[i] * 2.0f;
    }

    LOG.debug("Packet %d: e[0] = b[0] * 2 = %.1f", p.id, p.eVec[0]);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
