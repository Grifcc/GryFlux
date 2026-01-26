#include "CSubConstNode.h"

#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <chrono>
#include <random>
#include <thread>

namespace PipelineNodes
{

void CSubConstNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.cVec[i] = p.aVec[i] - 1.0f;
    }

    // random error injection for testing
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 99);
    if (dis(gen) == 1)
    {
        LOG.error("Packet %d: Injected error in node 'c_sub'", p.id);
        throw std::runtime_error("Simulated error in CSubConstNode");
    }

    LOG.debug("Packet %d: c[0] = a[0] - 1 = %.1f", p.id, p.cVec[0]);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
