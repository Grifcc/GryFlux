/*************************************************************************************************************************
 * GryFlux Framework - segmentation_deeplab_ascend Async Demo
 *************************************************************************************************************************/

#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "acl/acl.h"

#include "consumer/deeplab_result_consumer.h"
#include "context/deeplab_npu_Context.h"
#include "nodes/custom_nodes.h"
#include "source/deeplab_source.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void printUsage(const char *programName)
{
    std::cerr
        << "Usage: " << programName
        << " <image_dir> <label_dir> <model_path> [npu_instances] [thread_pool_size] [max_active_packets]\n";
}

void validateArgCount(int argc, char **argv)
{
    if (argc < 4)
    {
        static const char *kRequiredArgs[] = {"image_dir", "label_dir", "model_path"};
        const int missingIndex = argc - 1;
        const std::string missingArg =
            (missingIndex >= 0 && missingIndex < 3) ? kRequiredArgs[missingIndex] : "unknown";

        printUsage(argv[0]);
        throw std::runtime_error("missing required argument: " + missingArg);
    }

    if (argc > 7)
    {
        printUsage(argv[0]);
        throw std::runtime_error("too many arguments: unexpected extra parameter '" + std::string(argv[7]) + "'");
    }
}

void validatePathExists(const std::filesystem::path &path, const std::string &label)
{
    if (!std::filesystem::exists(path))
    {
        throw std::runtime_error("invalid " + label + ": path does not exist: " + path.string());
    }
}

size_t parsePositiveSizeArgument(const char *argValue, const std::string &label)
{
    const std::string value = (argValue != nullptr) ? std::string(argValue) : std::string();
    if (value.empty())
    {
        throw std::runtime_error("invalid " + label + ": value is empty");
    }

    size_t parsedChars = 0;
    unsigned long long parsedValue = 0;
    try
    {
        parsedValue = std::stoull(value, &parsedChars, 10);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error("invalid " + label + ": '" + value + "' is not a positive integer");
    }

    if (parsedChars != value.size())
    {
        throw std::runtime_error("invalid " + label + ": '" + value + "' contains non-numeric characters");
    }

    if (parsedValue == 0)
    {
        throw std::runtime_error("invalid " + label + ": value must be greater than 0");
    }

    if (parsedValue > static_cast<unsigned long long>(std::numeric_limits<size_t>::max()))
    {
        throw std::runtime_error("invalid " + label + ": value is too large");
    }

    return static_cast<size_t>(parsedValue);
}

} // namespace

int main(int argc, char **argv)
{
    bool aclInitialized = false;

    try
    {
        validateArgCount(argc, argv);

        const std::filesystem::path imageDir = argv[1];
        const std::filesystem::path labelDir = argv[2];
        const std::filesystem::path modelPath = argv[3];
        const size_t npuInstances = (argc >= 5) ? parsePositiveSizeArgument(argv[4], "npu_instances") : 1;
        const size_t threadPoolSize = (argc >= 6) ? parsePositiveSizeArgument(argv[5], "thread_pool_size") : 8;
        const size_t maxActivePackets = (argc >= 7) ? parsePositiveSizeArgument(argv[6], "max_active_packets") : 4;

        validatePathExists(imageDir, "image_dir");
        validatePathExists(labelDir, "label_dir");
        validatePathExists(modelPath, "model_path");

        LOG.setLevel(GryFlux::LogLevel::INFO);
        LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
        LOG.setAppName("deeplab");

        LOG.info("=======================================================");
        LOG.info("  GryFlux segmentation_deeplab_ascend Async Demo");
        LOG.info("=======================================================");
        LOG.info("Image dir : %s", imageDir.string().c_str());
        LOG.info("Label dir : %s", labelDir.string().c_str());
        LOG.info("Model path: %s", modelPath.string().c_str());

        const aclError aclRet = aclInit(nullptr);
        if (aclRet != ACL_SUCCESS)
        {
            throw std::runtime_error("aclInit failed with code: " + std::to_string(static_cast<int>(aclRet)));
        }
        aclInitialized = true;

        auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
        {
            std::vector<std::shared_ptr<GryFlux::Context>> npuContexts;
            npuContexts.reserve(npuInstances);
            for (size_t i = 0; i < npuInstances; ++i)
            {
                npuContexts.push_back(std::make_shared<DeepLabNPUContext>(
                    static_cast<int>(i),
                    modelPath.string()));
            }
            resourcePool->registerResourceType("npu", std::move(npuContexts));
        }
        LOG.info("Registered %zu NPU resources", npuInstances);

        auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
            [](GryFlux::TemplateBuilder *builder)
            {
                builder->setInputNode<PipelineNodes::InputNode>("source");
                builder->addTask<PipelineNodes::PreprocessNode>("preprocess", "", {"source"});
                builder->addTask<PipelineNodes::InferenceNode>("inference", "npu", {"preprocess"});
                builder->addTask<PipelineNodes::PostprocessNode>("postprocess", "", {"inference"});
                builder->addTask<PipelineNodes::GtProcessNode>("gt_process", "", {"source"});
                builder->setOutputNode<PipelineNodes::MiouNode>("miou", {"postprocess", "gt_process"});
            });

        auto source = std::make_shared<DeepLabSource>(imageDir.string(), labelDir.string());
        auto consumer = std::make_shared<DeepLabResultConsumer>(
            static_cast<size_t>(source->getTotalImages()));

        LOG.info("Discovered %d images", source->getTotalImages());
        LOG.info("Starting async pipeline with threadPoolSize=%zu, maxActivePackets=%zu",
                 threadPoolSize,
                 maxActivePackets);

        const auto startTime = std::chrono::steady_clock::now();

        {
            GryFlux::AsyncPipeline pipeline(
                source,
                graphTemplate,
                resourcePool,
                consumer,
                threadPoolSize,
                maxActivePackets);

            if constexpr (GryFlux::Profiling::kBuildProfiling)
            {
                pipeline.setProfilingEnabled(true);
            }

            pipeline.run();

            if constexpr (GryFlux::Profiling::kBuildProfiling)
            {
                pipeline.printProfilingStats();
                pipeline.dumpProfilingTimeline("deeplab_timeline.json");
            }
        }

        const auto endTime = std::chrono::steady_clock::now();
        const auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        consumer->printSummary();
        LOG.info("All packets completed in %lld ms", static_cast<long long>(durationMs.count()));

        resourcePool.reset();

        if (aclInitialized)
        {
            const aclError finalizeRet = aclFinalize();
            if (finalizeRet != ACL_SUCCESS)
            {
                LOG.error("aclFinalize failed with code: %d", static_cast<int>(finalizeRet));
                return 1;
            }
            aclInitialized = false;
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        LOG.error("DeepLab app failed: %s", e.what());

        if (aclInitialized)
        {
            const aclError finalizeRet = aclFinalize();
            if (finalizeRet != ACL_SUCCESS)
            {
                LOG.error("aclFinalize failed with code: %d", static_cast<int>(finalizeRet));
            }
        }

        return 1;
    }
}
