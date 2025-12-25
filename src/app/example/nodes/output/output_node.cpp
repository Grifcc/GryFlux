#include "output_node.h"

#include "packet/calc_packet.h"

namespace TestNodes
{

void OutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;

    // Output node is the final 'f' node in this test graph.
    auto &p = static_cast<CalcPacket &>(packet);
    const size_t n = p.f.size();
    for (size_t i = 0; i < n; ++i)
    {
        p.f[i] = p.d[i] + p.z[i];
    }
}

} // namespace TestNodes
