#include "IAddNode.h"

#include "context/AdderContext.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <vector>

namespace PipelineNodes
{

void IAddNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &adder = static_cast<AdderContext &>(ctx);

    // i = h + d
    adder.compute(p.iVec, p.hVec, p.dVec);

    LOG.debug(
        "Packet %d: i[0] = h[0] + d[0] = %.1f (adder=%d)",
        p.id,
        p.iVec[0],
        adder.getDeviceId());

}

} // namespace PipelineNodes
