/*************************************************************************************************************************
 * GryFlux Framework - DeepLab Async Demo
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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::filesystem::path defaultModelPath()
{
    return std::filesystem::path(__FILE__).parent_path() / "model" / "deeplabv3_int8.om";
}

void validatePathExists(const std::filesystem::path &path, const std::string &label)
{
    if (!std::filesystem::exists(path))
    {
        throw std::runtime_error(label + " does not exist: " + path.string());
    }
}

} // namespace

int main(int argc, char **argv)
{
    bool aclInitialized = false;

    try
    {
        if (argc < 3)
        {
            std::cerr
                << "Usage: " << argv[0]
                << " <image_dir> <label_dir> [model_path] [npu_instances] [thread_pool_size] [max_active_packets]\n";
            return 1;
        }

        const std::filesystem::path imageDir = argv[1];
        const std::filesystem::path labelDir = argv[2];
        const std::filesystem::path modelPath = (argc >= 4) ? std::filesystem::path(argv[3]) : defaultModelPath();
        const size_t npuInstances = (argc >= 5) ? static_cast<size_t>(std::stoul(argv[4])) : 1;
        const size_t threadPoolSize = (argc >= 6) ? static_cast<size_t>(std::stoul(argv[5])) : 8;
        const size_t maxActivePackets = (argc >= 7) ? static_cast<size_t>(std::stoul(argv[6])) : 4;

        validatePathExists(imageDir, "image_dir");
        validatePathExists(labelDir, "label_dir");
        validatePathExists(modelPath, "model_path");

        LOG.setLevel(GryFlux::LogLevel::INFO);
        LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
        LOG.setAppName("deeplab");

        LOG.info("========================================");
        LOG.info("  GryFlux DeepLab Async Demo");
        LOG.info("========================================");
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
