#include "BMulConstNode.h"

#include "context/MultiplierContext.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <vector>

namespace PipelineNodes
{

void BMulConstNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
	auto &p = static_cast<SimpleDataPacket &>(packet);
	auto &mul = static_cast<MultiplierContext &>(ctx);

	mul.compute(p.bVec, p.aVec, 2.0f);

	LOG.debug("Packet %d: b[0] = a[0] * 2 = %.1f (mul=%d)", p.id, p.bVec[0], mul.getDeviceId());
}

} // namespace PipelineNodes
