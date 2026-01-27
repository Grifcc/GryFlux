#include "FMulConstNode.h"

#include "context/MultiplierContext.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <vector>

namespace PipelineNodes
{

void FMulConstNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &mul = static_cast<MultiplierContext &>(ctx);

    mul.compute(p.fVec, p.bVec, 3.0f);

    LOG.debug("Packet %d: f[0] = b[0] * 3 = %.1f (mul=%d)", p.id, p.fVec[0], mul.getDeviceId());
}

} // namespace PipelineNodes
