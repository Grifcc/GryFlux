#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "consumer/deeplab_result_consumer.h"
#include "context/infercontext.h"
#include "nodes/custom_nodes.h"
#include "source/deeplab_source.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

namespace {

constexpr size_t kNpuInstances = 3;
constexpr size_t kThreadPoolSize = 16;
constexpr size_t kMaxActivePackets = 3;
constexpr const char kProfilingTimelinePath[] = "deeplab_timeline.json";

struct AppOptions {
    std::string model_path;
    std::string input_dir;
    std::string output_dir;
};

bool ParseAppOptions(int argc, char** argv, const char* app_name, AppOptions* options) {
    if (argc != 4) {
        std::cout << "Usage: " << app_name
                  << " <om_model_path> <input_dir> <output_dir>" << std::endl;
        return false;
    }

    options->model_path = argv[1];
    options->input_dir = argv[2];
    options->output_dir = argv[3];
    return true;
}

void InitializeLogger() {
    cv::setNumThreads(0);
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("deeplab");
}

std::shared_ptr<GryFlux::ResourcePool> BuildResourcePool(const AppOptions& options) {
    auto resource_pool = std::make_shared<GryFlux::ResourcePool>();
    resource_pool->registerResourceType(
        "npu",
        CreateInferContexts(options.model_path, 0, kNpuInstances));
    return resource_pool;
}

std::shared_ptr<GryFlux::GraphTemplate> BuildGraphTemplate(const AppOptions& options) {
    return GryFlux::GraphTemplate::buildOnce(
        [&options](GryFlux::TemplateBuilder* builder) {
            builder->setInputNode<PipelineNodes::InputNode>("source");
            builder->addTask<PipelineNodes::PreprocessNode>("preprocess", "", {"source"});
            builder->addTask<PipelineNodes::InferenceNode>("inference", "npu", {"preprocess"});
            builder->addTask<PipelineNodes::PostprocessNode>("postprocess", "", {"inference"});
            builder->setOutputNode<PipelineNodes::OutputNode>(
                "output",
                {"postprocess"},
                options.output_dir,
                10);
        });
}

int RunPipeline(const AppOptions& options) {
    LOG.info("=======================================================");
    LOG.info("  GryFlux segmentation_deeplab_ascend Async Demo");
    LOG.info("=======================================================");
    LOG.info("Image dir : %s", options.input_dir.c_str());
    LOG.info("Model path: %s", options.model_path.c_str());
    LOG.info("Output dir: %s", options.output_dir.c_str());

    auto resource_pool = BuildResourcePool(options);
    auto graph_template = BuildGraphTemplate(options);
    auto source = std::make_shared<DeepLabSource>(options.input_dir);
    if (source->getTotalImages() == 0) {
        throw std::runtime_error("未找到可处理的输入图片！");
    }

    auto consumer = std::make_shared<DeepLabResultConsumer>(
        static_cast<size_t>(source->getTotalImages()));

    LOG.info("Discovered %d images", source->getTotalImages());
    LOG.info("Starting async pipeline...");

    GryFlux::AsyncPipeline pipeline(
        source,
        graph_template,
        resource_pool,
        consumer,
        kThreadPoolSize,
        kMaxActivePackets);

    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        pipeline.setProfilingEnabled(true);
    }

    pipeline.run();

    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        pipeline.printProfilingStats();
        pipeline.dumpProfilingTimeline(kProfilingTimelinePath);
    }

    consumer->printSummary();
    LOG.info("程序结束。");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    InitializeLogger();

    AppOptions options;
    if (!ParseAppOptions(argc, argv, argv[0], &options)) {
        return 1;
    }

    try {
        return RunPipeline(options);
    } catch (const std::exception& exception) {
        LOG.error("segmentation_deeplab_ascend failed: %s", exception.what());
        return 1;
    }
}
