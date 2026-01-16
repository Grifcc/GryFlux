/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Parallel Pipeline Example
 *
 * This example demonstrates a DAG with parallel nodes:
 *
 * DAG Structure:
 *
 *   Input
 *     ├─→ ImagePreprocess ─→ FeatExtractor ─┐
 *     └─→ ObjectDetection ───────────────────→ ObjectTracker
 *
 * Data Flow (verifiable):
 * - SimpleDataSource:  Generates packets with id = 0, 1, 2, ...
 * - Input:             rawValue = id
 * - ImagePreprocess:   preprocessedValue = rawValue * 2        (并行分支1)
 * - ObjectDetection:   detectionValue = rawValue + 10          (并行分支2, NPU)
 * - FeatExtractor:     featureValue = preprocessedValue + 5
 * - ObjectTracker:     trackValue = detectionValue + featureValue
 * - ResultConsumer:    Verifies trackValue == (id + 10) + (id * 2 + 5) = 3 * id + 15
 *
 * Expected Results:
 * - Packet 0: track = 3 * 0 + 15 = 15.0
 * - Packet 1: track = 3 * 1 + 15 = 18.0
 * - Packet 2: track = 3 * 2 + 15 = 21.0
 * - Packet 3: track = 3 * 3 + 15 = 24.0
 * - ...
 *
 * 关键设计：
 * - ImagePreprocess 和 ObjectDetection 并行执行（都依赖 Input）
 * - 每个节点写入不同的字段，避免数据竞争
 * - ObjectTracker 是融合节点，等待两个前置节点完成
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
#include "nodes/dag/dag_nodes.h"

// Source and Consumer
#include "source/simple_data_source.h"
#include "consumer/result_consumer.h"

#include <iostream>
#include <chrono>
#include <queue>
#include <sstream>
#include <algorithm>

int main(int argc, char **argv)
{
    // Initialize logger
    LOG.setLevel(GryFlux::LogLevel::INFO);  // 改为 DEBUG 可查看 NPU 详细操作
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("SimplePipelineExample");

    LOG.info("========================================");
    LOG.info("  GryFlux DAG Example (new_example)");
    LOG.info("  6 layers, 10 nodes: a-j");
    LOG.info("========================================");

    // -------------------- Step 1: Create Resource Pool --------------------

    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();

    // Register limited resources: adder / multiplier
    resourcePool->registerResourceType("adder", {
                                                    std::make_shared<SimulatedAdderContext>(0),
                                                    std::make_shared<SimulatedAdderContext>(1)});
    LOG.info("Registered 2 Adder resources");

    // b/c can run in parallel, so provide 2 multiplier resources
    resourcePool->registerResourceType("multiplier", {
                                                         std::make_shared<SimulatedMultiplierContext>(0),
                                                         std::make_shared<SimulatedMultiplierContext>(1)});
    LOG.info("Registered 2 Multiplier resources");

    // -------------------- Step 2: Build Graph Template --------------------

    // Build graph template:
    // Layer 1: a
    // Layer 2: b, c, d (depend on a)
    // Layer 3: e, f, g (e<-b, f<-b, g<-b,c,d)
    // Layer 4: h (h<-e,f,g)
    // Layer 5: i (i<-h,c)
    // Layer 6: j (j<-i)
    // NOTE: d/e/f do not depend on any resource.
    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder *builder)
        {
            builder->setInputNode<PipelineNodes::ANode>("a");

            builder->addTask<PipelineNodes::BNode>("b", "multiplier", {"a"});
            builder->addTask<PipelineNodes::CNode>("c", "multiplier", {"a"});
            builder->addTask<PipelineNodes::DNode>("d", "", {"a"});

            builder->addTask<PipelineNodes::ENode>("e", "", {"b"});
            builder->addTask<PipelineNodes::FNode>("f", "multiplier", {"b"});
            builder->addTask<PipelineNodes::GNode>("g", "adder", {"b", "c", "d"});

            builder->addTask<PipelineNodes::HNode>("h", "adder", {"e", "f", "g"});

            builder->addTask<PipelineNodes::INode>("i", "adder", {"h", "c"});

            builder->setOutputNode<PipelineNodes::JNode>("j", {"i"});
        });

    LOG.info("Transformation: i = 19 * id + 3");

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
        12 // Thread pool size (default maxActivePackets = threadPoolSize - 1 = 11)
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

    // Theoretical max throughput (packets/sec), based on current delays and resources
    // Node counts by resource (current DAG):
    // CPU: a, d, e, j  => 4 nodes
    // adder: g, h, i   => 3 nodes
    // multiplier: b, c, f => 3 nodes
    constexpr double kCpuDelayMs = 60.0;
    constexpr double kAdderDelayMs = 50.0;
    constexpr double kMultiplierDelayMs = 100.0;
    constexpr double kThreadPoolSize = 12.0;
    constexpr double kAdderInstances = 2.0;
    constexpr double kMultiplierInstances = 2.0;

    const double cpuDemandMs = 4.0 * kCpuDelayMs + 3.0 * kAdderDelayMs + 3.0 * kMultiplierDelayMs;
    const double addDemandMs = 3.0 * kAdderDelayMs;
    const double mulDemandMs = 3.0 * kMultiplierDelayMs;

    const double cpuMaxThroughput = (kThreadPoolSize * 1000.0) / cpuDemandMs;
    const double addMaxThroughput = (kAdderInstances * 1000.0) / addDemandMs;
    const double mulMaxThroughput = (kMultiplierInstances * 1000.0) / mulDemandMs;

    const double theoreticalMaxThroughput = std::min({cpuMaxThroughput, addMaxThroughput, mulMaxThroughput});

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
