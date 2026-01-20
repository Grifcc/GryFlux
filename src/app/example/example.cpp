/*************************************************************************************************************************
 * GryFlux Framework - DAG Example 
 *
 * This example demonstrates a small DAG with limited resources (adder/multiplier)
 * plus CPU-only nodes.
 *
 * DAG Structure (6 layers, 10 nodes):
 *├
 *   input
 *     │               ─────────→ e_mul4 ──────────┐
 *     ├─→ b_mul2 ───→├─────────→ f_mul6 ──────→ hsum_efg ───→ isum_hc ──→ output
 *     │              └─────────→ gsum_bcd ────────┘             ↑
 *     ├─→ d_add3 ───────────────────┘                           │
 *     │                             ↑                           │
 *     └─→ c_mul3 ───────────────────┘───────────────────────────┘
 *
 * Data Flow (verifiable):
 * - input:     a = id
 * - b_mul2:    b = a * 2
 * - c_mul3:    c = a * 3
 * - d_add3:    d = a + 3
 * - e_mul4:    e = b * 2
 * - f_mul6:    f = b * 3
 * - gsum_bcd:  g = b + c + d
 * - hsum_efg:  h = e + f + g
 * - isum_hc:   i = h + c
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
#include "nodes/dag_nodes.h"

// Source and Consumer
#include "source/simple_data_source.h"
#include "consumer/result_consumer.h"

#include <algorithm>
#include <chrono>
#include <vector>

static double computeTheoreticalMaxThroughputPps(
    size_t threadPoolSize,
    size_t adderInstances,
    size_t multiplierInstances);

int main(int argc, char **argv)
{
    // Initialize logger
    LOG.setLevel(GryFlux::LogLevel::INFO);  // 改为 DEBUG 可查看各节点详细日志
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("new_example");

    LOG.info("========================================");
    LOG.info("  GryFlux DAG Example (new_example)");
    LOG.info("  6 layers, 10 nodes: input-output");
    LOG.info("========================================");

    // -------------------- Step 1: Create Resource Pool --------------------

    // Keep runtime config in one place so throughput estimation reads the same values.
    constexpr size_t kThreadPoolSize = 24;
    constexpr size_t kMaxActivePackets = 6;
    constexpr size_t kAdderInstances = 2;
    constexpr size_t kMultiplierInstances = 2;

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

    // mul2/mul3 can run in parallel, so provide multiple multiplier resources
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
    // Layer 2: b_mul2, c_mul3, d_add3 (depend on input)
    // Layer 3: e_mul4, f_mul6, gsum_bcd
    // Layer 4: hsum_efg
    // Layer 5: isum_hc
    // Layer 6: output
    // NOTE: CPU-only nodes do not depend on limited resources.
    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder *builder)
        {
            builder->setInputNode<PipelineNodes::InputNode>("input");

            builder->addTask<PipelineNodes::Mul2Node>("b_mul2", "multiplier", {"input"});
            builder->addTask<PipelineNodes::Mul3Node>("c_mul3", "multiplier", {"input"});
            builder->addTask<PipelineNodes::Add3Node>("d_add3", "", {"input"});

            builder->addTask<PipelineNodes::Mul4Node>("e_mul4", "", {"b_mul2"});
            builder->addTask<PipelineNodes::Mul6Node>("f_mul6", "multiplier", {"b_mul2"});
            builder->addTask<PipelineNodes::SumBcdNode>("gsum_bcd", "adder", {"b_mul2", "c_mul3", "d_add3"});

            builder->addTask<PipelineNodes::SumEfgNode>("hsum_efg", "adder", {"e_mul4", "f_mul6", "gsum_bcd"});

            builder->addTask<PipelineNodes::SumHcNode>("isum_hc", "adder", {"hsum_efg", "c_mul3"});

            builder->setOutputNode<PipelineNodes::OutputNode>("output", {"isum_hc"});
        });

    LOG.info("Transformation: i = 19 * id + 3, output(j) = i");

    // -------------------- Step 3: Create Data Source --------------------

    const int NUM_PACKETS = 300;
    auto source = std::make_shared<SimpleDataSource>(NUM_PACKETS);

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
        kAdderInstances,
        kMultiplierInstances);

    LOG.info("========================================");
    LOG.info("All %d packets completed in %lld ms", NUM_PACKETS, duration.count());
    LOG.info("Average: %.2f ms/packet", duration.count() / static_cast<double>(NUM_PACKETS));
    LOG.info("Theoretical Max Throughput: %.2f packets/sec", theoreticalMaxThroughput);
    LOG.info("Throughput: %.2f packets/sec", 1000.0 * NUM_PACKETS / duration.count());
    LOG.info("========================================");
    LOG.info("Verification Results:");
    LOG.info("  ✓ Success: %zu packets", consumer->getSuccessCount());
    LOG.info("  ✗ Failure: %zu packets", consumer->getFailureCount());
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
    return (consumer->getFailureCount() == 0) ? 0 : 1;
}


static double computeTheoreticalMaxThroughputPps(
    size_t threadPoolSize,
    size_t adderInstances,
    size_t multiplierInstances)
{
    // Node counts by resource (current DAG):
    // CPU: input, d_add3, e_mul4, output => 4 nodes
    // adder: gsum_bcd, hsum_efg, isum_hc => 3 nodes
    // multiplier: b_mul2, c_mul3, f_mul6 => 3 nodes
    constexpr double kCpuNodesPerPacket = 4.0;
    constexpr double kAdderNodesPerPacket = 3.0;
    constexpr double kMultiplierNodesPerPacket = 3.0;

    const double cpuDelayMs = static_cast<double>(PipelineNodes::DagNodeDelayConfig::kCpuDelayMs);
    const double adderDelayMs = static_cast<double>(PipelineNodes::DagNodeDelayConfig::kAdderDelayMs);
    const double multiplierDelayMs = static_cast<double>(PipelineNodes::DagNodeDelayConfig::kMultiplierDelayMs);

    const double totalWorkMs =
        kCpuNodesPerPacket * cpuDelayMs +
        kAdderNodesPerPacket * adderDelayMs +
        kMultiplierNodesPerPacket * multiplierDelayMs;

    const double adderDemandMs = kAdderNodesPerPacket * adderDelayMs;
    const double multiplierDemandMs = kMultiplierNodesPerPacket * multiplierDelayMs;

    const double cpuMax = (static_cast<double>(threadPoolSize) * 1000.0) / totalWorkMs;
    const double adderMax = (static_cast<double>(adderInstances) * 1000.0) / adderDemandMs;
    const double mulMax = (static_cast<double>(multiplierInstances) * 1000.0) / multiplierDemandMs;

    return std::min({cpuMax, adderMax, mulMax});
}
