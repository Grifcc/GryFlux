#include "DAddNode.h"

#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>

namespace PipelineNodes
{

void DAddNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.dVec[i] = p.aVec[i] + 3.0f;
    }

    LOG.debug("Packet %d: d[0] = a[0] + 3 = %.1f", p.id, p.dVec[0]);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
