/*************************************************************************************************************************
 * GryFlux Framework - DAG Example 
 *
 * This example demonstrates a small DAG with limited resources (adder/multiplier)
 * plus CPU-only nodes.
 *
 * DAG Structure: assests/chart.svg (6 layers, 10 nodes)
 *
 * Data Flow (verifiable):
 * - input:     a = id
 * - BMul:      b = a * 2
 * - CMul:      c = a * 3
 * - DAdd:      d = a + 3
 * - EMul:      e = b * 2
 * - FMul:      f = b * 3
 * - GSum:      g = b + c + d
 * - HSum:      h = e + f + g
 * - ISum:      i = h + c
 * - output:    j = i
 *************************************************************************************************************************/

#include "framework/resource_pool.h"
#include "framework/graph_template.h"
#include "framework/template_builder.h"
#include "framework/async_pipeline.h"
#include "framework/profiler/profiling_build_config.h"
#include "utils/logger.h"

// Custom types
#include "context/simulated_adder_context.h"
#include "context/simulated_multiplier_context.h"
#include "packet/simple_data_packet.h"

// Pipeline nodes 
#include "nodes/Input/InputNode.h"
#include "nodes/BMul/BMulNode.h"
#include "nodes/CMul/CMulNode.h"
#include "nodes/DAdd/DAddNode.h"
#include "nodes/EMul/EMulNode.h"
#include "nodes/FMul/FMulNode.h"
#include "nodes/GSum/GSumNode.h"
#include "nodes/HSum/HSumNode.h"
#include "nodes/ISum/ISumNode.h"
#include "nodes/Output/OutputNode.h"

// Source and Consumer
#include "source/simple_data_source.h"
#include "consumer/result_consumer.h"

#include <algorithm>
#include <chrono>
#include <vector>

static double computeTheoreticalMaxThroughputPps(
    size_t threadPoolSize,
    size_t producerTimeMs,
    size_t adderInstances,
    size_t multiplierInstances,
    double cpuDelayMs,
    double adderDelayMs,
    double multiplierDelayMs);

int main(int argc, char **argv)
{
    // Initialize logger
    LOG.setLevel(GryFlux::LogLevel::INFO);  // 改为 DEBUG 可查看各节点详细日志
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("example");

    LOG.info("========================================");
    LOG.info("  GryFlux DAG Example ");
    LOG.info("  6 layers, 10 nodes: input-output");
    LOG.info("========================================");

    // -------------------- Step 1: Create Resource Pool --------------------

    constexpr size_t kThreadPoolSize = 24;
    constexpr size_t kMaxActivePackets = 6;

    constexpr size_t kAdderInstances = 2;   // To register 2 adder resources
    constexpr size_t kMultiplierInstances = 2;  // To register 2 multiplier resources

    constexpr int kCpuDelayMs = 2;
    constexpr int kAdderDelayMs = 5;
    constexpr int kMultiplierDelayMs = 10;

    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();

    // Register limited resources: adder / multiplier
    {
        std::vector<std::shared_ptr<GryFlux::Context>> adderContexts;
        adderContexts.reserve(kAdderInstances);
        for (size_t i = 0; i < kAdderInstances; ++i)
        {
            adderContexts.push_back(std::make_shared<SimulatedAdderContext>(static_cast<int>(i)));
        }
        resourcePool->registerResourceType("adder", std::move(adderContexts));
    }
    LOG.info("Registered %zu Adder resources", kAdderInstances);

    {
        std::vector<std::shared_ptr<GryFlux::Context>> multiplierContexts;
        multiplierContexts.reserve(kMultiplierInstances);
        for (size_t i = 0; i < kMultiplierInstances; ++i)
        {
            multiplierContexts.push_back(std::make_shared<SimulatedMultiplierContext>(static_cast<int>(i)));
        }
        resourcePool->registerResourceType("multiplier", std::move(multiplierContexts));
    }
    LOG.info("Registered %zu Multiplier resources", kMultiplierInstances);

    // -------------------- Step 2: Build Graph Template --------------------

    // Build graph template:
    // Layer 1: input
    // Layer 2: b_mul, c_mul, d_add (depend on input)
    // Layer 3: e_mul, f_mul, g_sum
    // Layer 4: h_sum
    // Layer 5: i_sum
    // Layer 6: output
    // NOTE: CPU-only nodes do not depend on limited resources.
    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [=](GryFlux::TemplateBuilder *builder)
        {
            builder->setInputNode<PipelineNodes::InputNode>("input", kCpuDelayMs);
            builder->addTask<PipelineNodes::BMulNode>("b_mul", "multiplier", {"input"}, kMultiplierDelayMs);
            builder->addTask<PipelineNodes::CMulNode>("c_mul", "multiplier", {"input"}, kMultiplierDelayMs);
            builder->addTask<PipelineNodes::DAddNode>("d_add", "", {"input"}, kCpuDelayMs);

            builder->addTask<PipelineNodes::EMulNode>("e_mul", "", {"b_mul"}, kCpuDelayMs);
            builder->addTask<PipelineNodes::FMulNode>("f_mul", "multiplier", {"b_mul"}, kMultiplierDelayMs);
            builder->addTask<PipelineNodes::GSumNode>("g_sum", "adder", {"b_mul", "c_mul", "d_add"}, kAdderDelayMs);

            builder->addTask<PipelineNodes::HSumNode>("h_sum", "adder", {"e_mul", "f_mul", "g_sum"}, kAdderDelayMs);

            builder->addTask<PipelineNodes::ISumNode>("i_sum", "adder", {"h_sum", "c_mul"}, kAdderDelayMs);

            builder->setOutputNode<PipelineNodes::OutputNode>("output", {"i_sum"}, kCpuDelayMs);
        });

    LOG.info("Transformation: i = 19 * id + 3, output(j) = i");

    // -------------------- Step 3: Create Data Source --------------------

    constexpr int NUM_PACKETS = 300;
    constexpr int producerTimeMs = 16; // ms per packet
    auto source = std::make_shared<SimpleDataSource>(NUM_PACKETS, producerTimeMs);

    LOG.info("Created SimpleDataSource with %d packets", NUM_PACKETS);

    // -------------------- Step 4: Create Data Consumer --------------------

    auto consumer = std::make_shared<ResultConsumer>();

    LOG.info("Created ResultConsumer");

    // -------------------- Step 5: Create and Run Async Pipeline --------------------

    LOG.info("Starting async pipeline...");

    auto startTime = std::chrono::steady_clock::now();

    GryFlux::AsyncPipeline pipeline(
        source,
        graphTemplate,
        resourcePool,
        consumer,
        kThreadPoolSize,
        kMaxActivePackets
    );

    if constexpr (GryFlux::Profiling::kBuildProfiling)
    {
        LOG.info("Graph profiler 已启用（build-time: GRYFLUX_BUILD_PROFILING=1）");
        pipeline.setProfilingEnabled(true);
    }
    else
    {
        LOG.info("Graph profiler 未编译（build-time: GRYFLUX_BUILD_PROFILING=0），如需启用请使用 -DGRYFLUX_BUILD_PROFILING=1 重新编译。");
    }

    // Run pipeline (blocks until all frames processed)
    pipeline.run();

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // -------------------- Step 6: Show Statistics --------------------
    const double theoreticalMaxThroughput = computeTheoreticalMaxThroughputPps(
        kThreadPoolSize,
        producerTimeMs,
        kAdderInstances,
        kMultiplierInstances,
        static_cast<double>(kCpuDelayMs),
        static_cast<double>(kAdderDelayMs),
        static_cast<double>(kMultiplierDelayMs));

    LOG.info("========================================");
    LOG.info("All %d packets completed in %lld ms", NUM_PACKETS, duration.count());
    LOG.info("Average: %.2f ms/packet", duration.count() / static_cast<double>(NUM_PACKETS));
    LOG.info("Theoretical Max Throughput: %.2f packets/sec", theoreticalMaxThroughput);
    LOG.info("Throughput: %.2f packets/sec", 1000.0 * NUM_PACKETS / duration.count());
    LOG.info("========================================");
    LOG.info("Verification Results:");
    const size_t consumedPackets = consumer->getConsumedCount();
    const size_t droppedOrFailedPackets =
        (NUM_PACKETS >= static_cast<int>(consumedPackets))
            ? static_cast<size_t>(NUM_PACKETS) - consumedPackets
            : 0;

    LOG.info("  ✓ Consumed: %zu packets", consumedPackets);
    LOG.info("  ✗ Failure : %zu packets", droppedOrFailedPackets);
    LOG.info("========================================");

    // -------------------- Step 7: Show Profiling Statistics --------------------
    if constexpr (GryFlux::Profiling::kBuildProfiling)
    {
        LOG.info("Graph Profiler Summary:");
        pipeline.printProfilingStats();

        const std::string timelinePath = "graph_timeline.json";
        pipeline.dumpProfilingTimeline(timelinePath);
        LOG.info("Graph timeline dumped to %s (可用于可视化)", timelinePath.c_str());
    }
    else
    {
        LOG.info("Graph profiler 未编译，跳过统计输出。");
    }
    // return (consumer->getFailureCount() == 0) ? 0 : 1;
}


static double computeTheoreticalMaxThroughputPps(
    size_t threadPoolSize,
    size_t producerTimeMs,
    size_t adderInstances,
    size_t multiplierInstances,
    double cpuDelayMs,
    double adderDelayMs,
    double multiplierDelayMs)
{
    // Node counts by resource (current DAG):
    // CPU: input, d_add, e_mul, output => 4 nodes
    // adder: g_sum, h_sum, i_sum => 3 nodes
    // multiplier: b_mul, c_mul, f_mul => 3 nodes
    constexpr double kCpuNodesPerPacket = 4.0;
    constexpr double kAdderNodesPerPacket = 3.0;
    constexpr double kMultiplierNodesPerPacket = 3.0;

    const double totalWorkMs =
        kCpuNodesPerPacket * cpuDelayMs +
        kAdderNodesPerPacket * adderDelayMs +
        kMultiplierNodesPerPacket * multiplierDelayMs;

    const double adderDemandMs = kAdderNodesPerPacket * adderDelayMs;
    const double multiplierDemandMs = kMultiplierNodesPerPacket * multiplierDelayMs;

    const double producerCount = static_cast<double>(1000.0 / producerTimeMs);
    const double cpuMax = (static_cast<double>(threadPoolSize) * 1000.0) / totalWorkMs;
    const double adderMax = (static_cast<double>(adderInstances) * 1000.0) / adderDemandMs;
    const double mulMax = (static_cast<double>(multiplierInstances) * 1000.0) / multiplierDemandMs;

    return std::min({cpuMax, adderMax, mulMax, producerCount});
}
