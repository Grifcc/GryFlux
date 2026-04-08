#include "nodes/Input/InputNode.h"

#include "packet/deeplab_packet.h"

namespace DeeplabNodes
{

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeeplabPacket &>(packet);
    (void)p;
}

} // namespace DeeplabNodes
