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
#include "source/image_data_source.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

enum class ParseAppOptionsResult {
    kRun,
    kExitSuccess,
    kExitFailure,
};

struct AppOptions {
    std::string input_path;
    std::string output_path;
    std::string model_path;

    int model_width = 640;
    int model_height = 640;
    int device_id = 0;

    float confidence_threshold = 0.3F;
    float nms_threshold = 0.45F;

    size_t thread_pool_size = 12;
    size_t max_active_packets = 8;
    size_t npu_instance_count = 2;
};

void ThrowIfAclError(aclError error, const char* action) {
    if (error == ACL_SUCCESS) {
        return;
    }

    throw std::runtime_error(
        std::string(action) + " failed, aclError=" + std::to_string(error));
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

    AscendAclGuard(const AscendAclGuard&) = delete;
    AscendAclGuard& operator=(const AscendAclGuard&) = delete;

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
        } else if constexpr (std::is_same_v<T, float>) {
            *output = std::stof(value);
        } else if constexpr (std::is_same_v<T, size_t>) {
            *output = static_cast<size_t>(std::stoull(value));
        }
    } catch (const std::exception&) {
        std::cerr << app_name << ": invalid value for " << option_name << ": "
                  << value << std::endl;
        return false;
    }
    return true;
}

void PrintUsage(const char* app_name) {
    std::cout
        << "Usage: " << app_name << " [OPTIONS]\n"
        << "Options:\n"
        << "  -h, --help                  Show this help message\n"
        << "  -i, --input <path>          Input video path (required)\n"
        << "  -o, --output <path>         Output video path (required)\n"
        << "  -m, --model <path>          Ascend OM model path (required)\n"
        << "  -w, --width <num>           Model input width (default: 640)\n"
        << "  -H, --height <num>          Model input height (default: 640)\n"
        << "  -c, --conf <float>          Confidence threshold (default: 0.3)\n"
        << "  -n, --nms <float>           NMS threshold (default: 0.45)\n"
        << "  -d, --device <num>          Ascend device id (default: 0)\n"
        << "      --threads <num>         AsyncPipeline thread pool size "
           "(default: 12)\n"
        << "      --max-active <num>      AsyncPipeline max active packets "
           "(default: 8)\n"
        << "      --npu-instances <num>   ACL inference context count "
           "(default: 2)\n";
}

bool ValidateOptions(const char* app_name, const AppOptions& options) {
    if (options.input_path.empty() || options.output_path.empty() ||
        options.model_path.empty()) {
        std::cerr << app_name
                  << ": --input, --output and --model are required."
                  << std::endl;
        return false;
    }

    if (options.model_width <= 0 || options.model_height <= 0) {
        std::cerr << app_name
                  << ": --width and --height must be positive integers."
                  << std::endl;
        return false;
    }

    if (options.thread_pool_size == 0 || options.max_active_packets == 0 ||
        options.npu_instance_count == 0) {
        std::cerr << app_name
                  << ": --threads, --max-active and --npu-instances must be "
                     "greater than zero."
                  << std::endl;
        return false;
    }

    if (options.confidence_threshold <= 0.0F ||
        options.confidence_threshold > 1.0F ||
        options.nms_threshold <= 0.0F || options.nms_threshold > 1.0F) {
        std::cerr << app_name
                  << ": --conf and --nms must be in the range (0, 1]."
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

        auto require_value = [&](const std::string& option_name)
                                 -> const char* {
            if (index + 1 >= argc) {
                std::cerr << app_name << ": missing value for " << option_name
                          << std::endl;
                return nullptr;
            }
            return argv[++index];
        };

        if (argument == "-i" || argument == "--input") {
            const char* value = require_value(argument);
            if (!value) {
                return ParseAppOptionsResult::kExitFailure;
            }
            options->input_path = value;
            continue;
        }
        if (argument == "-o" || argument == "--output") {
            const char* value = require_value(argument);
            if (!value) {
                return ParseAppOptionsResult::kExitFailure;
            }
            options->output_path = value;
            continue;
        }
        if (argument == "-m" || argument == "--model") {
            const char* value = require_value(argument);
            if (!value) {
                return ParseAppOptionsResult::kExitFailure;
            }
            options->model_path = value;
            continue;
        }
        if (argument == "-w" || argument == "--width") {
            const char* value = require_value(argument);
            if (!value ||
                !ParseScalarArgument(app_name, argument, value,
                                     &options->model_width)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "-H" || argument == "--height") {
            const char* value = require_value(argument);
            if (!value ||
                !ParseScalarArgument(app_name, argument, value,
                                     &options->model_height)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "-c" || argument == "--conf") {
            const char* value = require_value(argument);
            if (!value ||
                !ParseScalarArgument(app_name, argument, value,
                                     &options->confidence_threshold)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "-n" || argument == "--nms") {
            const char* value = require_value(argument);
            if (!value ||
                !ParseScalarArgument(app_name, argument, value,
                                     &options->nms_threshold)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "-d" || argument == "--device") {
            const char* value = require_value(argument);
            if (!value ||
                !ParseScalarArgument(app_name, argument, value,
                                     &options->device_id)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--threads") {
            const char* value = require_value(argument);
            if (!value ||
                !ParseScalarArgument(app_name, argument, value,
                                     &options->thread_pool_size)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--max-active") {
            const char* value = require_value(argument);
            if (!value ||
                !ParseScalarArgument(app_name, argument, value,
                                     &options->max_active_packets)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--npu-instances") {
            const char* value = require_value(argument);
            if (!value ||
                !ParseScalarArgument(app_name, argument, value,
                                     &options->npu_instance_count)) {
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
    LOG.setAppName("yolox_detection_310p");
}

std::shared_ptr<GryFlux::ResourcePool> BuildResourcePool(
    const AppOptions& options) {
    auto resource_pool = std::make_shared<GryFlux::ResourcePool>();
    resource_pool->registerResourceType(
        "npu",
        CreateInferContexts(
            options.model_path,
            options.device_id,
            options.npu_instance_count));
    return resource_pool;
}

std::shared_ptr<GryFlux::GraphTemplate> BuildGraphTemplate(
    const AppOptions& options) {
    return GryFlux::GraphTemplate::buildOnce(
        [&](GryFlux::TemplateBuilder* builder) {
            builder->setInputNode<PipelineNodes::InputNode>("input");
            builder->addTask<PipelineNodes::PreprocessNode>(
                "preprocess",
                "",
                {"input"},
                options.model_width,
                options.model_height);
            builder->addTask<PipelineNodes::InferenceNode>(
                "inference",
                "npu",
                {"preprocess"});
            builder->addTask<PipelineNodes::PostprocessNode>(
                "postprocess",
                "",
                {"inference"},
                options.model_width,
                options.model_height,
                options.confidence_threshold,
                options.nms_threshold);
            builder->setOutputNode<PipelineNodes::OutputNode>(
                "output",
                {"postprocess"});
        });
}

int RunPipeline(const AppOptions& options) {
    AscendAclGuard acl_guard(options.device_id);

    auto resource_pool = BuildResourcePool(options);
    auto graph_template = BuildGraphTemplate(options);
    auto source = std::make_shared<ImageDataSource>(
        options.input_path,
        options.model_width,
        options.model_height);
    auto consumer = std::make_shared<ResultConsumer>(
        options.output_path,
        source->getFps(),
        source->getWidth(),
        source->getHeight());

    GryFlux::AsyncPipeline pipeline(
        source,
        graph_template,
        resource_pool,
        consumer,
        options.thread_pool_size,
        options.max_active_packets);

    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        pipeline.setProfilingEnabled(true);
    }

    LOG.info("Starting yolox_detection_310p pipeline");
    const auto start_time = std::chrono::steady_clock::now();
    pipeline.run();
    const auto end_time = std::chrono::steady_clock::now();

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    LOG.info("Pipeline completed in %lld ms",
             static_cast<long long>(elapsed_ms.count()));

    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        pipeline.printProfilingStats();
        pipeline.dumpProfilingTimeline("graph_timeline.json");
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
        "yolox_detection_310p",
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
        LOG.error("yolox_detection_310p failed: %s", exception.what());
        return 1;
    }
}
