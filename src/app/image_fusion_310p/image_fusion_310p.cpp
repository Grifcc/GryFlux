#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "acl/acl.h"

#include "consumer/result_consumer.h"
#include "context/infercontext.h"
#include "nodes/Inference/InferenceNode.h"
#include "nodes/Input/InputNode.h"
#include "nodes/Output/OutputNode.h"
#include "nodes/Postprocess/PostprocessNode.h"
#include "nodes/Preprocess/PreprocessNode.h"
#include "source/fusion_data_source.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

namespace {

enum class ParseAppOptionsResult {
    kRun,
    kExitSuccess,
    kExitFailure,
};

struct AppOptions {
    std::string vis_dir;
    std::string ir_dir;
    std::string output_dir;
    std::string model_path;

    int model_width = 640;
    int model_height = 480;
    int device_id = 0;

    size_t thread_pool_size = 8;
    size_t max_active_packets = 16;
    size_t npu_instance_count = 2;
};

void ThrowIfAclError(aclError error, const char* action) {
    if (error == ACL_SUCCESS) {
        return;
    }
    throw std::runtime_error(std::string(action) + " failed, aclError=" + std::to_string(error));
}

class AscendAclGuard {
public:
    explicit AscendAclGuard(int device_id)
        : device_id_(device_id) {
        ThrowIfAclError(aclInit(nullptr), "aclInit");
        initialized_ = true;
        ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");
        device_bound_ = true;
    }

    ~AscendAclGuard() {
        if (device_bound_) {
            aclrtResetDevice(device_id_);
        }
        if (initialized_) {
            aclFinalize();
        }
    }

private:
    int device_id_ = 0;
    bool initialized_ = false;
    bool device_bound_ = false;
};

template <typename T>
bool ParseScalarArgument(
    const char* app_name,
    const std::string& option_name,
    const std::string& value,
    T* output) {
    try {
        if constexpr (std::is_same_v<T, int>) {
            *output = std::stoi(value);
        } else if constexpr (std::is_same_v<T, size_t>) {
            *output = static_cast<size_t>(std::stoull(value));
        }
    } catch (const std::exception&) {
        std::cerr << app_name << ": invalid value for " << option_name
                  << ": " << value << std::endl;
        return false;
    }
    return true;
}

void PrintUsage(const char* app_name) {
    std::cout
        << "Usage: " << app_name << " [OPTIONS]\n"
        << "Options:\n"
        << "  -h, --help                   Show this help message\n"
        << "      --vis <dir>              Visible image directory (required)\n"
        << "      --ir <dir>               Infrared image directory (required)\n"
        << "      --output <dir>           Fused image output directory (required)\n"
        << "      --model <path>           Ascend OM model path (required)\n"
        << "      --threads <num>          AsyncPipeline thread pool size (default: 8)\n"
        << "      --packets <num>          AsyncPipeline max active packets (default: 16)\n"
        << "      --npu-instances <num>    ACL inference context count (default: 2)\n"
        << "      --device <num>           Ascend device id (default: 0)\n"
        << "      --width <num>            Model input width (default: 640)\n"
        << "      --height <num>           Model input height (default: 480)\n";
}

bool ValidateOptions(const char* app_name, const AppOptions& options) {
    if (options.vis_dir.empty() || options.ir_dir.empty() ||
        options.output_dir.empty() || options.model_path.empty()) {
        std::cerr << app_name
                  << ": --vis, --ir, --output and --model are required."
                  << std::endl;
        return false;
    }
    if (options.model_width <= 0 || options.model_height <= 0 ||
        options.thread_pool_size == 0 || options.max_active_packets == 0 ||
        options.npu_instance_count == 0) {
        std::cerr << app_name
                  << ": dimensions, thread counts and instance counts must be greater than zero."
                  << std::endl;
        return false;
    }
    return true;
}

ParseAppOptionsResult ParseAppOptions(
    int argc,
    char** argv,
    const char* app_name,
    AppOptions* options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            PrintUsage(app_name);
            return ParseAppOptionsResult::kExitSuccess;
        }

        auto require_value = [&](const std::string& option_name) -> const char* {
            if (index + 1 >= argc) {
                std::cerr << app_name << ": missing value for " << option_name << std::endl;
                return nullptr;
            }
            return argv[++index];
        };

        if (argument == "--vis") {
            const char* value = require_value(argument);
            if (!value) return ParseAppOptionsResult::kExitFailure;
            options->vis_dir = value;
            continue;
        }
        if (argument == "--ir") {
            const char* value = require_value(argument);
            if (!value) return ParseAppOptionsResult::kExitFailure;
            options->ir_dir = value;
            continue;
        }
        if (argument == "--output") {
            const char* value = require_value(argument);
            if (!value) return ParseAppOptionsResult::kExitFailure;
            options->output_dir = value;
            continue;
        }
        if (argument == "--model") {
            const char* value = require_value(argument);
            if (!value) return ParseAppOptionsResult::kExitFailure;
            options->model_path = value;
            continue;
        }
        if (argument == "--threads") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->thread_pool_size)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--packets") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->max_active_packets)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--npu-instances") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->npu_instance_count)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--device") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->device_id)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--width") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->model_width)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--height") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->model_height)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }

        std::cerr << app_name << ": unknown option: " << argument << std::endl;
        return ParseAppOptionsResult::kExitFailure;
    }

    if (!ValidateOptions(app_name, *options)) {
        PrintUsage(app_name);
        return ParseAppOptionsResult::kExitFailure;
    }
    return ParseAppOptionsResult::kRun;
}

void InitializeLogger() {
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("image_fusion_310p");
}

struct AppRuntimeResources {
    FusionModelInfo model_info;
    std::shared_ptr<GryFlux::ResourcePool> resource_pool;
};

AppRuntimeResources BuildResourcePool(const AppOptions& options) {
    auto infer_bundle = CreateFusionInferResourceBundle(
        options.model_path,
        options.device_id,
        options.npu_instance_count,
        options.model_width,
        options.model_height);
    auto resource_pool = std::make_shared<GryFlux::ResourcePool>();
    resource_pool->registerResourceType("npu", std::move(infer_bundle.contexts));
    return AppRuntimeResources{infer_bundle.model_info, resource_pool};
}

std::shared_ptr<GryFlux::GraphTemplate> BuildGraphTemplate(
    const FusionModelInfo& model_info) {
    return GryFlux::GraphTemplate::buildOnce(
        [&](GryFlux::TemplateBuilder* builder) {
            builder->setInputNode<PipelineNodes::InputNode>("input");
            builder->addTask<PipelineNodes::PreprocessNode>(
                "preprocess",
                "",
                {"input"},
                model_info.model_width,
                model_info.model_height);
            builder->addTask<PipelineNodes::InferenceNode>(
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
    AscendAclGuard acl_guard(options.device_id);

    const AppRuntimeResources runtime_resources = BuildResourcePool(options);
    const auto graph_template = BuildGraphTemplate(runtime_resources.model_info);
    auto source = std::make_shared<FusionDataSource>(
        options.vis_dir,
        options.ir_dir,
        runtime_resources.model_info.model_width,
        runtime_resources.model_info.model_height);
    auto consumer = std::make_shared<ResultConsumer>(options.output_dir);

    GryFlux::AsyncPipeline pipeline(
        source,
        graph_template,
        runtime_resources.resource_pool,
        consumer,
        options.thread_pool_size,
        options.max_active_packets);

    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        pipeline.setProfilingEnabled(true);
    }

    LOG.info("Starting image_fusion_310p pipeline with model input %dx%d",
             runtime_resources.model_info.model_width,
             runtime_resources.model_info.model_height);
    const auto start_time = std::chrono::steady_clock::now();
    pipeline.run();
    const auto end_time = std::chrono::steady_clock::now();

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    LOG.info("image_fusion_310p completed in %lld ms",
             static_cast<long long>(elapsed_ms.count()));

    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        pipeline.printProfilingStats();
        pipeline.dumpProfilingTimeline("image_fusion_timeline.json");
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    InitializeLogger();

    AppOptions options;
    const auto parse_result = ParseAppOptions(
        argc,
        argv,
        "image_fusion_310p",
        &options);
    if (parse_result == ParseAppOptionsResult::kExitSuccess) {
        return 0;
    }
    if (parse_result == ParseAppOptionsResult::kExitFailure) {
        return 1;
    }

    try {
        return RunPipeline(options);
    } catch (const std::exception& exception) {
        LOG.error("image_fusion_310p failed: %s", exception.what());
        return 1;
    }
}
