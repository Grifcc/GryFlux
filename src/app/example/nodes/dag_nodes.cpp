#include "dag_nodes.h"

#include "packet/simple_data_packet.h"
#include "context/simulated_adder_context.h"
#include "context/simulated_multiplier_context.h"
#include "utils/logger.h"

#include <chrono>
#include <thread>

namespace PipelineNodes
{

namespace
{
inline std::chrono::milliseconds cpuDelay()
{
    return std::chrono::milliseconds(DagNodeDelayConfig::kCpuDelayMs);
}

inline std::chrono::milliseconds adderDelay()
{
    return std::chrono::milliseconds(DagNodeDelayConfig::kAdderDelayMs);
}

inline std::chrono::milliseconds multiplierDelay()
{
    return std::chrono::milliseconds(DagNodeDelayConfig::kMultiplierDelayMs);
}
} // namespace

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    const float x = static_cast<float>(p.id);
    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.aVec[i] = x;
    }

    LOG.debug("Packet %d: a[0] = %.1f", p.id, p.aVec[0]);

    std::this_thread::sleep_for(cpuDelay());
}

void Mul2Node::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &mul = static_cast<SimulatedMultiplierContext &>(ctx);

    // mul2: multiply on multiplier resource
    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.bVec[i] = mul.mul(p.aVec[i], 2.0f);
    }

    LOG.debug("Packet %d: b[0] = a[0] * 2 = %.1f (mul=%d)", p.id, p.bVec[0], mul.getDeviceId());

    std::this_thread::sleep_for(multiplierDelay());
}

void Mul3Node::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &mul = static_cast<SimulatedMultiplierContext &>(ctx);

    // mul3: multiply on multiplier resource
    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.cVec[i] = mul.mul(p.aVec[i], 3.0f);
    }

    LOG.debug("Packet %d: c[0] = a[0] * 3 = %.1f (mul=%d)", p.id, p.cVec[0], mul.getDeviceId());

    std::this_thread::sleep_for(multiplierDelay());
}

void Add3Node::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.dVec[i] = p.aVec[i] + 3.0f;
    }

    LOG.debug("Packet %d: d[0] = a[0] + 3 = %.1f", p.id, p.dVec[0]);

    std::this_thread::sleep_for(cpuDelay());
}

void Mul4Node::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.eVec[i] = p.bVec[i] * 2.0f;
    }

    LOG.debug("Packet %d: e[0] = b[0] * 2 = %.1f", p.id, p.eVec[0]);

    std::this_thread::sleep_for(cpuDelay());
}

void Mul6Node::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &mul = static_cast<SimulatedMultiplierContext &>(ctx);

    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.fVec[i] = mul.mul(p.bVec[i], 3.0f);
    }

    LOG.debug("Packet %d: f[0] = b[0] * 3 = %.1f (mul=%d)", p.id, p.fVec[0], mul.getDeviceId());

    std::this_thread::sleep_for(multiplierDelay());
}

void SumBcdNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &adder = static_cast<SimulatedAdderContext &>(ctx);

    // g depends on b,c,d and uses adder resource:
    // g = b + c + d
    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.gVec[i] = adder.add(adder.add(p.bVec[i], p.cVec[i]), p.dVec[i]);
    }
    if (p.id % 10 == 0)
    {
        LOG.error("Packet %d: Injected error in node 'gsum_bcd'", p.id);
        packet.markFailed();
        return;
    }

    LOG.debug("Packet %d: g[0] = b[0] + c[0] + d[0] = %.1f (adder=%d)", p.id, p.gVec[0], adder.getDeviceId());

    std::this_thread::sleep_for(adderDelay());
}

void SumEfgNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &adder = static_cast<SimulatedAdderContext &>(ctx);

    // h = e + f + g (use adder resource)
    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.hVec[i] = adder.add(adder.add(p.eVec[i], p.fVec[i]), p.gVec[i]);
    }

    LOG.debug("Packet %d: h[0] = e[0] + f[0] + g[0] = %.1f (adder=%d)", p.id, p.hVec[0], adder.getDeviceId());

    std::this_thread::sleep_for(adderDelay());
}

void SumHcNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    auto &adder = static_cast<SimulatedAdderContext &>(ctx);

    // sum_hc depends on h and c.
    for (size_t i = 0; i < SimpleDataPacket::kVecSize; ++i)
    {
        p.iVec[i] = adder.add(p.hVec[i], p.cVec[i]);
    }

    LOG.debug("Packet %d: i[0] = h[0] + c[0] = %.1f (adder=%d)", p.id, p.iVec[0], adder.getDeviceId());

    std::this_thread::sleep_for(adderDelay());
}

void OutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<SimpleDataPacket &>(packet);

    // Final output node: pass-through
    p.jVec = p.iVec;

    LOG.debug("Packet %d: j[0] = i[0] = %.1f", p.id, p.jVec[0]);

    std::this_thread::sleep_for(cpuDelay());
}

} // namespace PipelineNodes
