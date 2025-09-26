#include <atomic>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "framework/processing_task.h"
#include "framework/streaming_pipeline.h"
#include "package.h"
#include "utils/unified_allocator.h"
#include "runtime/rknn_api.h"
#include "sink/write_consumer/write_consumer.h"
#include "source/producer/image_producer.h"
#include "source/producer/video_producer.h"
#include "tasks/deepsort_tracker/deepsort_tracker.h"
#include "tasks/image_preprocess/image_preprocess.h"
#include "tasks/object_detector/object_detector.h"
#include "tasks/rk_runner/rk_runner.h"
#include "tasks/track_result_sender/track_result_sender.h"
#include "utils/logger.h"

namespace
{
    void initLogger()
    {
        LOG.setLevel(GryFlux::LogLevel::INFO);
        LOG.setOutputType(GryFlux::LogOutputType::BOTH);
        LOG.setAppName("YoloxDeepSortStream");
        std::filesystem::create_directories("./logs");
        LOG.setLogFileRoot("./logs");
    }

    void buildGraph(std::shared_ptr<GryFlux::PipelineBuilder> builder,
                    std::shared_ptr<GryFlux::DataObject> input,
                    const std::string &outputId,
                    GryFlux::TaskRegistry &registry)
    {
        auto inputNode = builder->addInput("input", input);
        auto preprocessNode = builder->addTask("imagePreprocess", registry.getProcessFunction("imagePreprocess"), {inputNode});
        auto rkRunnerNode = builder->addTask("rkRunner", registry.getProcessFunction("rkRunner"), {preprocessNode});
        auto detectorNode = builder->addTask("objectDetector",
                                             registry.getProcessFunction("objectDetector"),
                                             {preprocessNode, rkRunnerNode});
        auto trackerNode = builder->addTask("deepSortTracker",
                                            registry.getProcessFunction("deepSortTracker"),
                                            {inputNode, detectorNode});
        builder->addTask(outputId,
                         registry.getProcessFunction("resultSender"),
                         {inputNode, trackerNode});
    }
}

int main(int argc, const char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <yolox_model> <reid_model> <video_or_image_path>" << std::endl;
        return 1;
    }

    initLogger();

    GryFlux::TaskRegistry registry;
    registry.registerTask<GryFlux::ImagePreprocess>("imagePreprocess", 640, 640);
    registry.registerTask<GryFlux::RkRunner>("rkRunner", argv[1], 1, 640, 640);
    registry.registerTask<GryFlux::ObjectDetector>("objectDetector", 0.5f);
    registry.registerTask<GryFlux::DeepSortTracker>("deepSortTracker", argv[2], 2, RKNN_NPU_CORE_2, 1, 512);
    registry.registerTask<GryFlux::TrackResultSender>("resultSender");

    GryFlux::StreamingPipeline pipeline(10);
    pipeline.setOutputNodeId("resultSender");
    pipeline.enableProfiling(true);

    pipeline.setProcessor([&registry](std::shared_ptr<GryFlux::PipelineBuilder> builder,
                                      std::shared_ptr<GryFlux::DataObject> input,
                                      const std::string &outputId) {
        buildGraph(builder, input, outputId, registry);
    });

    pipeline.start();

    std::atomic<bool> running(true);
    CPUAllocator *cpuAllocator = new CPUAllocator();

    std::unique_ptr<GryFlux::DataProducer> producer;
    const std::string input_path = argv[3];
    if (std::filesystem::is_directory(input_path))
    {
        producer = std::make_unique<GryFlux::ImageProducer>(pipeline, running, cpuAllocator, input_path);
    }
    else
    {
        producer = std::make_unique<GryFlux::VideoProducer>(pipeline, running, cpuAllocator, input_path);
    }

    const std::string outputDir = "./outputs";
    const std::string outputVideo = outputDir + "/result.mp4";
    constexpr double outputFps = 25.0;

    GryFlux::WriteConsumer consumer(pipeline, running, cpuAllocator, outputDir, outputVideo, outputFps);

    producer->start();
    consumer.start();

    producer->join();
    running.store(false);
    consumer.join();

    pipeline.stop();

    delete cpuAllocator;
    LOG.info("Pipeline finished");
    return 0;
}
