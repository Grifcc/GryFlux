#include "input_node.h"

#include "packet/calc_packet.h"

namespace TestNodes
{

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<CalcPacket &>(packet);
    const double v = static_cast<double>(p.id);
    for (auto &e : p.x)
    {
        e = v;
    }
}

} // namespace TestNodes
