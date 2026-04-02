#include "nodes/Output/OutputNode.h"

#include "packet/fusionnetv2_packet.h"

namespace FusionNetV2Nodes
{

void OutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<FusionNetV2Packet &>(packet);
    (void)p;
}

} // namespace FusionNetV2Nodes
