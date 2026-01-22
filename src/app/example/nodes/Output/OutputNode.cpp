#include "OutputNode.h"

#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>

namespace PipelineNodes
{

void OutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    // Final output node: pass-through
    p.jVec = p.iVec;

    LOG.debug("Packet %d: j[0] = i[0] = %.1f", p.id, p.jVec[0]);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
