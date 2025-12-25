#include "add_nodes.h"

#include "context/adder_context.h"

#include <chrono>
#include <thread>

namespace TestNodes
{

void AddConstNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    if (sleepMs_ > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs_));
    }
    auto &p = static_cast<CalcPacket &>(packet);
    auto &adder = static_cast<AdderContext &>(ctx);
    auto &in = p.*in_;
    auto &out = p.*out_;
    adder.add(in, addend_, out);
}

void Add2Node::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    if (sleepMs_ > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs_));
    }
    auto &p = static_cast<CalcPacket &>(packet);
    auto &adder = static_cast<AdderContext &>(ctx);
    auto &in1 = p.*in1_;
    auto &in2 = p.*in2_;
    auto &out = p.*out_;
    adder.add(in1, in2, out);
    // Keep the second add call to simulate extra compute, but avoid temp allocations.
    adder.add(out, 0.0, out);
}

void Fuse3SumNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    if (sleepMs_ > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs_));
    }
    auto &p = static_cast<CalcPacket &>(packet);
    auto &adder = static_cast<AdderContext &>(ctx);
    auto &in1 = p.*in1_;
    auto &in2 = p.*in2_;
    auto &in3 = p.*in3_;
    auto &out = p.*out_;
    adder.add(in1, in2, out);
    adder.add(out, in3, out);
}

} // namespace TestNodes
