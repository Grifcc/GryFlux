#include "FMulNode.h"

#include "context/simulated_multiplier_context.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>

namespace PipelineNodes
{

void FMulNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &mul = static_cast<SimulatedMultiplierContext &>(ctx);

    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.fVec[i] = mul.mul(p.bVec[i], 3.0f);
    }

    LOG.debug("Packet %d: f[0] = b[0] * 3 = %.1f (mul=%d)", p.id, p.fVec[0], mul.getDeviceId());

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
