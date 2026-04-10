#include "nodes/Input/InputNode.h"

#include "packet/resnet_packet.h"

namespace ResnetNodes
{

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<ResnetPacket &>(packet);
    (void)p;
}

} // namespace ResnetNodes

