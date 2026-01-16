#include "dag_nodes.h"

#include "packet/simple_data_packet.h"
#include "context/simulated_adder_context.h"
#include "context/simulated_multiplier_context.h"
#include "utils/logger.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace PipelineNodes
{

namespace
{
constexpr auto kCpuNodeDelay = std::chrono::milliseconds(60);

constexpr auto kAdderNodeDelay = std::chrono::milliseconds(50);
constexpr auto kMultiplierNodeDelay = std::chrono::milliseconds(100);
} // namespace

void ANode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    p.aValue = static_cast<float>(p.id);

    LOG.debug("Packet %d: a = %.1f", p.id, p.aValue);

    std::this_thread::sleep_for(kCpuNodeDelay);
}

void BNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &mul = static_cast<SimulatedMultiplierContext &>(ctx);

    // b: multiply on multiplier resource
    p.bValue = mul.mul(p.aValue, 2.0f);

    LOG.debug("Packet %d: b = a * 2 = %.1f (mul=%d)", p.id, p.bValue, mul.getDeviceId());

    std::this_thread::sleep_for(kMultiplierNodeDelay);
}

void CNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &mul = static_cast<SimulatedMultiplierContext &>(ctx);

    // c: multiply on multiplier resource
    p.cValue = mul.mul(p.aValue, 3.0f);

    LOG.debug("Packet %d: c = a * 3 = %.1f (mul=%d)", p.id, p.cValue, mul.getDeviceId());

    std::this_thread::sleep_for(kMultiplierNodeDelay);
}

void DNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    p.dValue = p.aValue + 3.0f;

    LOG.debug("Packet %d: d = a + 3 = %.1f", p.id, p.dValue);

    std::this_thread::sleep_for(kCpuNodeDelay);
}

void ENode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    p.eValue = p.bValue * 2.0f;

    LOG.debug("Packet %d: e = b * 2 = %.1f", p.id, p.eValue);

    std::this_thread::sleep_for(kCpuNodeDelay);
}

void FNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &mul = static_cast<SimulatedMultiplierContext &>(ctx);

    p.fValue = mul.mul(p.bValue, 3.0f);

    LOG.debug("Packet %d: f = b * 3 = %.1f (mul=%d)", p.id, p.fValue, mul.getDeviceId());

    std::this_thread::sleep_for(kMultiplierNodeDelay);
}

void GNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    // Fault injection: for ids that are multiples of 10, do a divide-by-zero and throw.
    // Use floating-point divide-by-zero (does not SIGFPE on most platforms) and then
    // throw an exception so GryFlux can catch it and mark the packet failed.
    if (p.id % 10 == 0)
    {
        volatile float denom = 0.0f;
        volatile float value = 1.0f / denom; // intentional divide by zero
        (void)value;

        LOG.error("Packet %d: Injected divide-by-zero in node 'g'", p.id);
        throw std::runtime_error("Injected divide-by-zero");
    }

    auto &adder = static_cast<SimulatedAdderContext &>(ctx);

    // g depends on b,c,d and uses adder resource:
    // g = b + c + d
    p.gValue = adder.add(adder.add(p.bValue, p.cValue), p.dValue);

    LOG.debug("Packet %d: g = b + c + d = %.1f (adder=%d)", p.id, p.gValue, adder.getDeviceId());

    std::this_thread::sleep_for(kAdderNodeDelay);
}

void HNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &adder = static_cast<SimulatedAdderContext &>(ctx);

    // h = e + f + g (use adder resource)
    p.hValue = adder.add(adder.add(p.eValue, p.fValue), p.gValue);

    LOG.debug("Packet %d: h = e + f + g = %.1f (adder=%d)", p.id, p.hValue, adder.getDeviceId());

    std::this_thread::sleep_for(kAdderNodeDelay);
}

void INode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &adder = static_cast<SimulatedAdderContext &>(ctx);

    // Output node i depends on h and c.
    p.iValue = adder.add(p.hValue, p.cValue);

    LOG.debug("Packet %d: i = h + c = %.1f (adder=%d)", p.id, p.iValue, adder.getDeviceId());

    std::this_thread::sleep_for(kAdderNodeDelay);
}

void JNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    // Final output node: pass-through
    p.jValue = p.iValue;

    LOG.debug("Packet %d: j = i = %.1f", p.id, p.jValue);

    std::this_thread::sleep_for(kCpuNodeDelay);
}

} // namespace PipelineNodes
