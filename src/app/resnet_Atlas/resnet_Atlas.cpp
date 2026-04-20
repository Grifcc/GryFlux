#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "context/acl_infer_context.h"


#include "consumer/resnet_result_consumer.h"
#include "nodes/Infer/InferNode.h"
#include "nodes/Input/InputNode.h"
#include "nodes/Output/OutputNode.h"
#include "nodes/Postprocess/PostprocessNode.h"
#include "nodes/Preprocess/PreprocessNode.h"
#include "source/resnet_data_source.h"

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

enum class ParseResult {
    kRun,
    kExitSuccess,
    kExitFailure,
};

constexpr size_t kThreadPoolSize = 8;
constexpr size_t kMaxActivePackets = 16;
constexpr size_t kNpuInstanceCount = 2;
constexpr const char kProfilingTimelinePath[] = "graph_timeline_resnet.json";

struct AppOptions {
    std::string model_path;
    std::string dataset_dir;
    std::string gt_file_path;
};

void PrintUsage(const char* app_name) {
    std::cout
        << "Usage: " << app_name << " [OPTIONS]\n"
        << "Options:\n"
        << "  -h, --help                  Show this help message\n"
        << "  -m, --model <path>          Ascend OM model path (required)\n"
        << "  -d, --dataset <path>        Dataset directory (required)\n"
        << "  -g, --gt <path>             Ground-truth label file (required)\n";
}

bool ValidateOptions(const char* app_name, const AppOptions& options) {
    if (options.model_path.empty() || options.dataset_dir.empty() ||
        options.gt_file_path.empty()) {
        std::cerr << app_name
                  << ": --model, --dataset and --gt are required."
                  << std::endl;
        return false;
    }
    return true;
}

ParseResult ParseAppOptions(int argc,
                            char** argv,
                            const char* app_name,
                            AppOptions* options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "-h" || argument == "--help") {
            PrintUsage(app_name);
            return ParseResult::kExitSuccess;
        }

        auto require_value = [&](const std::string& option_name)
                                 -> const char* {
            if (index + 1 >= argc) {
                std::cerr << app_name << ": missing value for " << option_name
                          << std::endl;
                return nullptr;
            }
            return argv[++index];
        };

        if (argument == "-m" || argument == "--model") {
            const char* value = require_value(argument);
            if (!value) {
                return ParseResult::kExitFailure;
            }
            options->model_path = value;
            continue;
        }
        if (argument == "-d" || argument == "--dataset") {
            const char* value = require_value(argument);
            if (!value) {
                return ParseResult::kExitFailure;
            }
            options->dataset_dir = value;
            continue;
        }
        if (argument == "-g" || argument == "--gt") {
            const char* value = require_value(argument);
            if (!value) {
                return ParseResult::kExitFailure;
            }
            options->gt_file_path = value;
            continue;
        }
        std::cerr << app_name << ": unknown option: " << argument << std::endl;
        return ParseResult::kExitFailure;
    }

    if (!ValidateOptions(app_name, *options)) {
        PrintUsage(app_name);
        return ParseResult::kExitFailure;
    }

    return ParseResult::kRun;
}

void InitializeLogger() {
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("resnet_Atlas");
}

std::map<std::string, int> LoadGroundTruth(const std::string& gt_file_path) {
    std::map<std::string, int> gt_map;
    std::ifstream ifs(gt_file_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open ground-truth file: " + gt_file_path);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        const size_t split_pos = line.find(' ');
        if (split_pos == std::string::npos) {
            continue;
        }
        const std::string filename = line.substr(0, split_pos);
        const int label_index = std::stoi(line.substr(split_pos + 1));
        gt_map[filename] = label_index;
    }
    return gt_map;
}

std::shared_ptr<GryFlux::ResourcePool> BuildResourcePool(
    const AppOptions& options) {
    auto resource_pool = std::make_shared<GryFlux::ResourcePool>();
    resource_pool->registerResourceType(
        "npu",
        resnet::CreateAclInferContexts(
            options.model_path,
            0,
            kNpuInstanceCount));
    return resource_pool;
}

std::shared_ptr<GryFlux::GraphTemplate> BuildGraphTemplate() {
    return GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder* builder) {
            builder->setInputNode<PipelineNodes::InputNode>("input");
            builder->addTask<PipelineNodes::PreprocessNode>(
                "preprocess",
                "",
                {"input"});
            builder->addTask<PipelineNodes::InferNode>(
                "inference",
                "npu",
                {"preprocess"});
            builder->addTask<PipelineNodes::PostprocessNode>(
                "postprocess",
                "",
                {"inference"});
            builder->setOutputNode<PipelineNodes::OutputNode>(
                "output",
                {"postprocess"});
        });
}

int RunPipeline(const AppOptions& options) {
    const auto gt_map = LoadGroundTruth(options.gt_file_path);
    LOG.info("Loaded %zu ground-truth entries from %s",
             gt_map.size(),
             options.gt_file_path.c_str());

    auto resource_pool = BuildResourcePool(options);
    auto graph_template = BuildGraphTemplate();
    auto source = std::make_shared<ResNetDataSource>(options.dataset_dir, gt_map);
    auto consumer = std::make_shared<ResNetResultConsumer>(gt_map.size());

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

    LOG.info("Starting resnet_Atlas pipeline");
    const auto start_time = std::chrono::steady_clock::now();
    pipeline.run();
    const auto end_time = std::chrono::steady_clock::now();

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    LOG.info("Pipeline completed in %lld ms",
             static_cast<long long>(elapsed_ms.count()));

    consumer->printMetrics();

    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        LOG.info("Profiling enabled, dumping statistics and timeline to %s",
                 kProfilingTimelinePath);
        pipeline.printProfilingStats();
        pipeline.dumpProfilingTimeline(kProfilingTimelinePath);
        LOG.info("Profiling timeline written to %s", kProfilingTimelinePath);
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    InitializeLogger();

    AppOptions options;
    const ParseResult parse_result =
        ParseAppOptions(argc, argv, "resnet_Atlas", &options);
    if (parse_result == ParseResult::kExitSuccess) {
        return 0;
    }
    if (parse_result == ParseResult::kExitFailure) {
        return 1;
    }

    try {
        return RunPipeline(options);
    } catch (const std::exception& exception) {
        LOG.error("resnet_Atlas failed: %s", exception.what());
        return 1;
    }
}
