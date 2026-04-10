#include "nodes/Input/InputNode.h"

#include "packet/realesrgan_packet.h"

namespace RealesrganNodes
{

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<RealesrganPacket &>(packet);
    (void)p;
}

} // namespace RealesrganNodes
