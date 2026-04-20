#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "consumer/DiskWriter/AsyncDiskWriter.h"
#include "consumer/ResultConsumer/ZeroDceResultConsumer.h"
#include "context/orin_context.h"
#include "context/trt_model_handle.h"
#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "nodes/Infer/InferNode.h"
#include "nodes/Postprocess/PostprocessNode.h"
#include "nodes/Preprocess/PreprocessNode.h"
#include "source/ZeroDceDataSource.h"

namespace {

constexpr size_t kThreadPoolSize = 8;
constexpr size_t kMaxActivePackets = 16;
constexpr size_t kOrinContextInstances = 2;
constexpr int kOrinDeviceId = 0;
constexpr const char* kOrinResourceType = "orin_trt";

struct AppConfig {
    std::string engine_path;
    std::string input_dir;
    std::string output_dir;
    std::string gt_dir;
    bool enable_save = true;
    bool enable_metrics = true;
    bool infer_only = false;
};

void PrintUsage(const char* program_name) {
    std::cout << "用法: " << program_name
              << " <engine_path> <input_dir> <output_dir> [gt_dir] [--no-save] [--no-metrics] [--infer-only]"
              << std::endl;
    std::cout << "示例:\n"
              << "  cd build\n"
              << "  ./src/app/ZeroDCE_Orin/zero_dce_app "
              << "/workspace/zjx/model/ZeroDCE_static640_int8.engine "
              << "/workspace/zjx/data/DICM "
              << "/workspace/zjx/out\n"
              << "  ./src/app/ZeroDCE_Orin/zero_dce_app "
              << "/workspace/zjx/model/ZeroDCE_static640_int8.engine "
              << "/workspace/zjx/data/DICM "
              << "/workspace/zjx/out_bench --infer-only" << std::endl;
}

bool ParseArgs(int argc, char* argv[], AppConfig* config) {
    if (argc < 4 || config == nullptr) {
        return false;
    }

    config->engine_path = argv[1];
    config->input_dir = argv[2];
    config->output_dir = argv[3];
    config->gt_dir.clear();
    config->enable_save = true;
    config->enable_metrics = true;
    config->infer_only = false;

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-save") {
            config->enable_save = false;
        } else if (arg == "--no-metrics") {
            config->enable_metrics = false;
        } else if (arg == "--infer-only") {
            config->infer_only = true;
        } else if (config->gt_dir.empty()) {
            config->gt_dir = arg;
        } else {
            return false;
        }
    }

    if (config->infer_only) {
        config->enable_save = false;
        config->enable_metrics = false;
    }
    return true;
}

std::shared_ptr<GryFlux::ResourcePool> CreateResourcePool(
    const std::shared_ptr<TrtModelHandle>& model_handle) {
    auto resource_pool = std::make_shared<GryFlux::ResourcePool>();

    std::vector<std::shared_ptr<GryFlux::Context>> orin_contexts;
    orin_contexts.reserve(kOrinContextInstances);
    for (size_t i = 0; i < kOrinContextInstances; ++i) {
        orin_contexts.push_back(std::make_shared<OrinContext>(kOrinDeviceId, model_handle));
    }

    resource_pool->registerResourceType(kOrinResourceType, std::move(orin_contexts));
    return resource_pool;
}

std::shared_ptr<GryFlux::GraphTemplate> BuildGraphTemplate() {
    return GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder* builder) {
            builder->setInputNode<PreprocessNode>("preprocess");
            builder->addTask<InferNode>("inference", kOrinResourceType, {"preprocess"});
            builder->setOutputNode<PostprocessNode>("postprocess", {"inference"});
        });
}

int RunApp(const AppConfig& config) {
    cv::setNumThreads(0);

    if (config.enable_save) {
        AsyncDiskWriter::GetInstance().Start(config.output_dir);
    }

    auto model_handle = std::make_shared<TrtModelHandle>(kOrinDeviceId, config.engine_path);
    auto source = std::make_shared<ZeroDceDataSource>(
        config.input_dir,
        config.gt_dir,
        config.enable_save,
        config.enable_metrics,
        config.infer_only,
        model_handle->inputChannels(),
        model_handle->inputHeight(),
        model_handle->inputWidth(),
        model_handle->outputChannels(),
        model_handle->outputHeight(),
        model_handle->outputWidth());
    auto consumer = std::make_shared<ZeroDceResultConsumer>(
        source->GetTotalFrames(),
        !config.gt_dir.empty(),
        config.enable_metrics,
        config.infer_only);
    auto resource_pool = CreateResourcePool(model_handle);
    auto graph_template = BuildGraphTemplate();

    GryFlux::AsyncPipeline pipeline(
        source,
        graph_template,
        resource_pool,
        consumer,
        kThreadPoolSize,
        kMaxActivePackets);

    auto finish_signal = consumer->get_future();
    std::exception_ptr pipeline_error;
    std::thread pipeline_thread([&]() {
        try {
            pipeline.run();
        } catch (...) {
            pipeline_error = std::current_exception();
            consumer->signalFailure();
        }
    });

    finish_signal.wait();

    if (pipeline_thread.joinable()) {
        pipeline_thread.join();
    }

    if (config.enable_save) {
        AsyncDiskWriter::GetInstance().Stop();
    }

    if (pipeline_error) {
        std::rethrow_exception(pipeline_error);
    }

    consumer->printMetrics();
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    AppConfig config;
    if (!ParseArgs(argc, argv, &config)) {
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        return RunApp(config);
    } catch (const std::exception& ex) {
        if (config.enable_save) {
            AsyncDiskWriter::GetInstance().Stop();
        }
        std::cerr << "[ERROR] ZeroDCE_Orin 启动失败: " << ex.what() << std::endl;
        return 1;
    }
}
