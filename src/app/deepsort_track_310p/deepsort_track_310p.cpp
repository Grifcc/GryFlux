#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "acl/acl.h"

#include "consumer/result_consumer.h"
#include "context/infercontext.h"
#include "context/reid_context.h"
#include "nodes/DetectionInference/DetectionInferenceNode.h"
#include "nodes/Input/InputNode.h"
#include "nodes/Output/OutputNode.h"
#include "nodes/Postprocess/PostprocessNode.h"
#include "nodes/Preprocess/PreprocessNode.h"
#include "nodes/ReidInference/ReidInferenceNode.h"
#include "nodes/ReidPreprocess/ReidPreprocessNode.h"
#include "source/image_data_source.h"

#include <chrono>
#include <cstddef>
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
    std::string input_path;
    std::string output_path;
    std::string detection_model_path;
    std::string reid_model_path;

    int detection_model_width = 640;
    int detection_model_height = 640;
    int reid_width = 128;
    int reid_height = 256;
    int reid_feature_dim = 512;
    int device_id = 0;

    float confidence_threshold = 0.3F;
    float nms_threshold = 0.45F;

    size_t thread_pool_size = 12;
    size_t max_active_packets = 8;
    size_t detection_instance_count = 2;
    size_t reid_instance_count = 2;
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
        << "  -h, --help                    Show this help message\n"
        << "  -i, --input <path>            Input video path (required)\n"
        << "  -o, --output <path>           Output video path (required)\n"
        << "  -y, --yolox <path>            YOLOX OM model path (required)\n"
        << "  -r, --reid <path>             ReID OM model path (required)\n"
        << "  -t, --threads <num>           AsyncPipeline thread pool size (default: 12)\n"
        << "  -p, --packets <num>           AsyncPipeline max active packets (default: 8)\n"
        << "      --det-width <num>         Detection model input width (default: 640)\n"
        << "      --det-height <num>        Detection model input height (default: 640)\n"
        << "      --reid-width <num>        ReID crop width (default: 128)\n"
        << "      --reid-height <num>       ReID crop height (default: 256)\n"
        << "      --reid-feature-dim <num>  ReID feature dimension (default: 512)\n"
        << "      --conf <float>            Detection confidence threshold (default: 0.3)\n"
        << "      --nms <float>             Detection NMS threshold (default: 0.45)\n"
        << "      --det-npu-instances <num> Detection ACL context count (default: 2)\n"
        << "      --reid-npu-instances <num> ReID ACL context count (default: 2)\n"
        << "  -d, --device <num>            Ascend device id (default: 0)\n";
}

bool ValidateOptions(const char* app_name, const AppOptions& options) {
    if (options.input_path.empty() || options.output_path.empty() ||
        options.detection_model_path.empty() || options.reid_model_path.empty()) {
        std::cerr << app_name
                  << ": --input, --output, --yolox and --reid are required."
                  << std::endl;
        return false;
    }

    if (options.detection_model_width <= 0 || options.detection_model_height <= 0 ||
        options.reid_width <= 0 || options.reid_height <= 0 ||
        options.reid_feature_dim <= 0) {
        std::cerr << app_name
                  << ": all model and feature dimensions must be positive."
                  << std::endl;
        return false;
    }

    if (options.thread_pool_size == 0 || options.max_active_packets == 0 ||
        options.detection_instance_count == 0 || options.reid_instance_count == 0) {
        std::cerr << app_name
                  << ": --threads, --packets, --det-npu-instances and --reid-npu-instances must be greater than zero."
                  << std::endl;
        return false;
    }

    if (options.confidence_threshold <= 0.0F || options.confidence_threshold > 1.0F ||
        options.nms_threshold <= 0.0F || options.nms_threshold > 1.0F) {
        std::cerr << app_name << ": --conf and --nms must be in the range (0, 1]."
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
        if (argument == "-y" || argument == "--yolox") {
            const char* value = require_value(argument);
            if (!value) {
                return ParseAppOptionsResult::kExitFailure;
            }
            options->detection_model_path = value;
            continue;
        }
        if (argument == "-r" || argument == "--reid") {
            const char* value = require_value(argument);
            if (!value) {
                return ParseAppOptionsResult::kExitFailure;
            }
            options->reid_model_path = value;
            continue;
        }
        if (argument == "-t" || argument == "--threads") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->thread_pool_size)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "-p" || argument == "--packets") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->max_active_packets)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--det-width") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->detection_model_width)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--det-height") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->detection_model_height)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--reid-width") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->reid_width)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--reid-height") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->reid_height)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--reid-feature-dim") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->reid_feature_dim)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--conf") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->confidence_threshold)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--nms") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->nms_threshold)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--det-npu-instances") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->detection_instance_count)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "--reid-npu-instances") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->reid_instance_count)) {
                return ParseAppOptionsResult::kExitFailure;
            }
            continue;
        }
        if (argument == "-d" || argument == "--device") {
            const char* value = require_value(argument);
            if (!value || !ParseScalarArgument(app_name, argument, value, &options->device_id)) {
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
    LOG.setAppName("deepsort_track_310p");
}

std::shared_ptr<GryFlux::ResourcePool> BuildResourcePool(
    const AppOptions& options) {
    auto resource_pool = std::make_shared<GryFlux::ResourcePool>();
    resource_pool->registerResourceType(
        "detector_npu",
        CreateDetectionInferContexts(
            options.detection_model_path,
            options.device_id,
            options.detection_instance_count));
    resource_pool->registerResourceType(
        "reid_npu",
        CreateReidInferContexts(
            options.reid_model_path,
            options.device_id,
            options.reid_instance_count));
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
                options.detection_model_width,
                options.detection_model_height);
            builder->addTask<PipelineNodes::DetectionInferenceNode>(
                "detection_inference",
                "detector_npu",
                {"preprocess"});
            builder->addTask<PipelineNodes::PostprocessNode>(
                "postprocess",
                "",
                {"detection_inference"},
                options.detection_model_width,
                options.detection_model_height,
                options.confidence_threshold,
                options.nms_threshold);
            builder->addTask<PipelineNodes::ReidPreprocessNode>(
                "reid_preprocess",
                "",
                {"postprocess"},
                options.reid_width,
                options.reid_height);
            builder->addTask<PipelineNodes::ReidInferenceNode>(
                "reid_inference",
                "reid_npu",
                {"reid_preprocess"},
                options.reid_feature_dim);
            builder->setOutputNode<PipelineNodes::OutputNode>(
                "output",
                {"reid_inference"});
        });
}

int RunPipeline(const AppOptions& options) {
    AscendAclGuard acl_guard(options.device_id);

    auto resource_pool = BuildResourcePool(options);
    auto graph_template = BuildGraphTemplate(options);
    auto source = std::make_shared<ImageDataSource>(
        options.input_path,
        options.detection_model_width,
        options.detection_model_height,
        options.reid_width,
        options.reid_height,
        options.reid_feature_dim);
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

    LOG.info("Starting deepsort_track_310p pipeline");
    const auto start_time = std::chrono::steady_clock::now();
    pipeline.run();
    const auto end_time = std::chrono::steady_clock::now();

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    LOG.info("Pipeline completed in %lld ms",
             static_cast<long long>(elapsed_ms.count()));

    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        pipeline.printProfilingStats();
        pipeline.dumpProfilingTimeline("deepsort_timeline.json");
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
        "deepsort_track_310p",
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
        LOG.error("deepsort_track_310p failed: %s", exception.what());
        return 1;
    }
}
