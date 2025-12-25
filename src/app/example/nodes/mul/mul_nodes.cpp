#include "mul_nodes.h"

#include "context/multiplier_context.h"

#include <chrono>
#include <thread>

namespace TestNodes
{

void MulConstNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    if (sleepMs_ > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs_));
    }
    auto &p = static_cast<CalcPacket &>(packet);
    auto &mul = static_cast<MultiplierContext &>(ctx);
    auto &in = p.*in_;
    auto &out = p.*out_;
    mul.mul(in, factor_, out);
}

} // namespace TestNodes
