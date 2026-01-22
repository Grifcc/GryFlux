#include "GSumNode.h"

#include "context/simulated_adder_context.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>
#include <random>
namespace PipelineNodes
{

void GSumNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &adder = static_cast<SimulatedAdderContext &>(ctx);

    // g = b + c + d
    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.gVec[i] = adder.add(adder.add(p.bVec[i], p.cVec[i]), p.dVec[i]);
    }
    // random error injection for testing
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 9);
    if (dis(gen) <= 1)
    {
        LOG.error("Packet %d: Injected error in node 'g_sum'", p.id);
        throw std::runtime_error("Simulated error in GSumNode");
    }

    LOG.debug(
        "Packet %d: g[0] = b[0] + c[0] + d[0] = %.1f (adder=%d)",
        p.id,
        p.gVec[0],
        adder.getDeviceId());

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
