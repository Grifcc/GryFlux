#include "GAddNode.h"

#include "context/AdderContext.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <random>
#include <vector>

namespace PipelineNodes
{

void GAddNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &adder = static_cast<AdderContext &>(ctx);

    // random error injection for testing
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 49);
    if (dis(gen) == 1)
    {
        LOG.error("Packet %d: Injected error in node 'g_add'", p.id);
        std::vector<float> empty;
        adder.compute(p.gVec, empty, empty);
    }
    else
    {
        // g = b + c + d
        adder.compute(p.gVec, p.bVec, p.cVec, p.dVec);
    }

    LOG.debug(
        "Packet %d: g[0] = b[0] + c[0] + d[0] = %.1f (adder=%d)",
        p.id,
        p.gVec[0],
        adder.getDeviceId());

}

} // namespace PipelineNodes
