#include "nodes/Input/InputNode.h"

#include "packet/fusionnetv2_packet.h"

namespace FusionNetV2Nodes
{

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<FusionNetV2Packet &>(packet);
    (void)p;
}

} // namespace FusionNetV2Nodes
