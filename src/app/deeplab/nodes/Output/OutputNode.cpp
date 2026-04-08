#include "nodes/Output/OutputNode.h"

#include "packet/deeplab_packet.h"

namespace DeeplabNodes
{

void OutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeeplabPacket &>(packet);
    (void)p;
}

} // namespace DeeplabNodes
