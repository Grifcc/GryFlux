#include "OutputNode.h"

#include "packet/realesrgan_packet.h"

namespace PipelineNodes
{

void OutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    (void)static_cast<RealEsrganPacket &>(packet);
}

} // namespace PipelineNodes
