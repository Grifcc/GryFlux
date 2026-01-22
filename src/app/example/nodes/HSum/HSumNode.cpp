#include "HSumNode.h"

#include "context/simulated_adder_context.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>

namespace PipelineNodes
{

void HSumNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &adder = static_cast<SimulatedAdderContext &>(ctx);

    // h = e + f + g
    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.hVec[i] = adder.add(adder.add(p.eVec[i], p.fVec[i]), p.gVec[i]);
    }

    LOG.debug(
        "Packet %d: h[0] = e[0] + f[0] + g[0] = %.1f (adder=%d)",
        p.id,
        p.hVec[0],
        adder.getDeviceId());

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
