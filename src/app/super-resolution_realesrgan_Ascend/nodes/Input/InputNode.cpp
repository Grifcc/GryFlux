#include "InputNode.h"

#include "packet/realesrgan_packet.h"

namespace PipelineNodes
{

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    (void)static_cast<RealEsrganPacket &>(packet);
}

} // namespace PipelineNodes
