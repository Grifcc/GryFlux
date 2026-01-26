#include "DMulConstNode.h"

#include "context/MultiplierContext.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <vector>

namespace PipelineNodes
{

void DMulConstNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &mul = static_cast<MultiplierContext &>(ctx);

    mul.compute(p.dVec, p.aVec, 4.0f);

    LOG.debug("Packet %d: d[0] = a[0] * 4 = %.1f (mul=%d)", p.id, p.dVec[0], mul.getDeviceId());
}

} // namespace PipelineNodes