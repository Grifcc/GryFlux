#include "HAddNode.h"

#include "context/AdderContext.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <vector>

namespace PipelineNodes
{

void HAddNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &adder = static_cast<AdderContext &>(ctx);

    // h = e + f + g
    adder.compute(p.hVec, p.eVec, p.fVec, p.gVec);

    LOG.debug(
        "Packet %d: h[0] = e[0] + f[0] + g[0] = %.1f (adder=%d)",
        p.id,
        p.hVec[0],
        adder.getDeviceId());

}

} // namespace PipelineNodes
