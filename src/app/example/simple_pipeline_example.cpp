/*************************************************************************************************************************
 * Copyright 2025 Grifcc
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
#include "framework/streaming_pipeline.h"
#include "utils/logger.h"

// Custom types
#include "context/simulated_npu_context.h"
#include "packet/simple_data_packet.h"

// Pipeline nodes
#include "nodes/input/input_node.h"
#include "nodes/preprocess/preprocess_node.h"
#include "nodes/inference/inference_node.h"
#include "nodes/postprocess/postprocess_node.h"
#include "nodes/output/output_node.h"

// Source and Consumer
#include "source/simple_data_source.h"
#include "consumer/result_consumer.h"

#include <iostream>
#include <chrono>

int main(int argc, char **argv)
{
    // Initialize logger
    LOG.setLevel(GryFlux::LogLevel::INFO);  // 改为 DEBUG 可查看 NPU 详细操作
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("SimplePipelineExample");

    LOG.info("========================================");
    LOG.info("  GryFlux Parallel Pipeline Example");
    LOG.info("  Demonstrates Parallel Node Execution");
    LOG.info("========================================");

    // -------------------- Step 1: Create Resource Pool --------------------

    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();

    // Register 2 simulated NPUs
    resourcePool->registerResourceType("npu", {
                                                   std::make_shared<SimulatedNPUContext>(0),
                                                   std::make_shared<SimulatedNPUContext>(1)});

    LOG.info("Registered 2 NPU resources");

    // -------------------- Step 2: Build Graph Template --------------------

    // Build graph template with parallel nodes
    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder *builder)
        {
            // Input Node - Entry point
            builder->setInputNode<PipelineNodes::InputNode>("input");

            // 并行分支 1: ImagePreprocess (CPU task)
            builder->addTask<PipelineNodes::ImagePreprocessNode>(
                "imagePreprocess",
                "",        // No resource needed (CPU task)
                {"input"}  // Depends on input node
            );

            // 并行分支 2: ObjectDetection (NPU task)
            builder->addTask<PipelineNodes::ObjectDetectionNode>(
                "objectDetection",
                "npu",     // Requires NPU resource
                {"input"}  // Depends on input node (并行!)
            );

            // FeatExtractor - Depends on ImagePreprocess
            builder->addTask<PipelineNodes::FeatExtractorNode>(
                "featExtractor",
                "",                     // No resource needed (CPU task)
                {"imagePreprocess"}     // Depends on imagePreprocess
            );

            // ObjectTracker - 融合节点 (depends on both branches)
            builder->setOutputNode<PipelineNodes::ObjectTrackerNode>(
                "objectTracker",
                {"objectDetection", "featExtractor"}  // Depends on both!
            );
        });

    LOG.info("Graph template built with parallel nodes:");
    LOG.info("  Input -> ImagePreprocess -> FeatExtractor ┐");
    LOG.info("       └-> ObjectDetection(NPU) ─────────────→ ObjectTracker");
    LOG.info("Transformation: track = (id + 10) + (id * 2 + 5) = 3 * id + 15");

    // -------------------- Step 3: Create Data Source --------------------

    const int NUM_PACKETS = 100;
    auto source = std::make_shared<SimpleDataSource>(NUM_PACKETS);

    LOG.info("Created SimpleDataSource with %d packets", NUM_PACKETS);

    // -------------------- Step 4: Create Data Consumer --------------------

    auto consumer = std::make_shared<ResultConsumer>();

    LOG.info("Created ResultConsumer");

    // -------------------- Step 5: Create and Run Streaming Pipeline --------------------

    LOG.info("Starting streaming pipeline...");

    auto startTime = std::chrono::steady_clock::now();

    GryFlux::StreamingPipeline pipeline(
        source,
        graphTemplate,
        resourcePool,
        consumer,
        8  // Thread pool size (maxActivePackets = 8 × 2 = 16 by default)
    );

    // Run pipeline (blocks until all frames processed)
    pipeline.run();

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // -------------------- Step 6: Show Statistics --------------------

    LOG.info("========================================");
    LOG.info("All %d packets completed in %lld ms", NUM_PACKETS, duration.count());
    LOG.info("Average: %.2f ms/packet", duration.count() / static_cast<double>(NUM_PACKETS));
    LOG.info("Throughput: %.2f packets/sec", 1000.0 * NUM_PACKETS / duration.count());
    LOG.info("========================================");
    LOG.info("Verification Results:");
    LOG.info("  ✓ Success: %zu packets", consumer->getSuccessCount());
    LOG.info("  ✗ Failure: %zu packets", consumer->getFailureCount());
    LOG.info("========================================");

    // -------------------- Step 7: Show Profiling Statistics --------------------

    LOG.info("Node Performance Statistics:");
    pipeline.printProfilingStats();

    return (consumer->getFailureCount() == 0) ? 0 : 1;
}
