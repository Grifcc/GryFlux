#include "nodes/Output/OutputNode.h"

#include "packet/resnet_packet.h"

namespace ResnetNodes
{

void OutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<ResnetPacket &>(packet);
    (void)p;
}

} // namespace ResnetNodes

