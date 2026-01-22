#include "CMulNode.h"

#include "context/simulated_multiplier_context.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>

namespace PipelineNodes
{

void CMulNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &mul = static_cast<SimulatedMultiplierContext &>(ctx);

    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.cVec[i] = mul.mul(p.aVec[i], 3.0f);
    }

    LOG.debug("Packet %d: c[0] = a[0] * 3 = %.1f (mul=%d)", p.id, p.cVec[0], mul.getDeviceId());

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
