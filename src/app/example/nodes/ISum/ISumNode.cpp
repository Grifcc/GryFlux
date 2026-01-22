#include "ISumNode.h"

#include "context/simulated_adder_context.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>

namespace PipelineNodes
{

void ISumNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &adder = static_cast<SimulatedAdderContext &>(ctx);

    // i = h + c
    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.iVec[i] = adder.add(p.hVec[i], p.cVec[i]);
    }

    LOG.debug(
        "Packet %d: i[0] = h[0] + c[0] = %.1f (adder=%d)",
        p.id,
        p.iVec[0],
        adder.getDeviceId());

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
