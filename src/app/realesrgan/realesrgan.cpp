#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "consumer/image_write_consumer.h"
#include "context/npu_context.h"
#include "nodes/realesrgan_nodes.h"
#include "source/image_dir_source.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
void printHelp()
{
    LOG.info("Usage: realesrgan <model_path> <dataset_dir> [output_dir] [options]");
    LOG.info("Options:");
    LOG.info("  --profile            Enable GryFlux profiling");
    LOG.info("  --help,-h            Show help and exit");
}
} // namespace

int main(int argc, char **argv)
{
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("realesrgan");

    if (argc < 3)
    {
        printHelp();
        return -1;
    }

    const std::string modelPath = argv[1];
    const std::string datasetPath = argv[2];

    std::string outputPath = "./outputs";
    int argIndex = 3;
    if (argIndex < argc && std::strncmp(argv[argIndex], "--", 2) != 0)
    {
        outputPath = argv[argIndex];
        ++argIndex;
    }

    bool enableProfiling = false;

    constexpr size_t kNpuInstances = 3;
    constexpr size_t kThreadPoolSize = 10;
    constexpr size_t kMaxActivePackets = 8;

    for (int i = argIndex; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h"))
        {
            printHelp();
            return 0;
        }
        if (!std::strcmp(argv[i], "--profile"))
        {
            enableProfiling = true;
            continue;
        }
        LOG.error("Unsupported option: %s", argv[i]);
        printHelp();
        return -1;
    }

    constexpr int kModelWidth = 256;
    constexpr int kModelHeight = 256;

    try
    {
        LOG.info("========================================");
        LOG.info("GryFlux RealESRGAN Pipeline");
        LOG.info("Model : %s", modelPath.c_str());
        LOG.info("Input : %s", datasetPath.c_str());
        LOG.info("Output: %s", outputPath.c_str());
        LOG.info("========================================");

        auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
        {
            std::vector<std::shared_ptr<GryFlux::Context>> npuContexts;
            npuContexts.reserve(kNpuInstances);
            for (size_t i = 0; i < kNpuInstances; ++i)
            {
                npuContexts.push_back(std::make_shared<NpuContext>(
                    static_cast<int>(i),
                    modelPath,
                    kModelWidth,
                    kModelHeight));
            }
            resourcePool->registerResourceType("npu", std::move(npuContexts));
        }

        auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
            [&](GryFlux::TemplateBuilder *builder)
            {
                builder->setInputNode<RealesrganNodes::InputNode>("input");
                builder->addTask<RealesrganNodes::PreprocessNode>("preprocess", "", {"input"}, kModelWidth, kModelHeight);
                builder->addTask<RealesrganNodes::InferenceNode>("inference", "npu", {"preprocess"});
                builder->addTask<RealesrganNodes::PostprocessNode>("postprocess", "", {"inference"});
                builder->setOutputNode<RealesrganNodes::OutputNode>("output", {"postprocess"});
            });

        auto source = std::make_shared<ImageDirSource>(datasetPath);
        auto consumer = std::make_shared<ImageWriteConsumer>(outputPath);

        GryFlux::AsyncPipeline pipeline(
            source,
            graphTemplate,
            resourcePool,
            consumer,
            kThreadPoolSize,
            kMaxActivePackets);

        if (enableProfiling)
        {
            if constexpr (GryFlux::Profiling::kBuildProfiling)
            {
                pipeline.setProfilingEnabled(true);
            }
            else
            {
                LOG.info("Graph profiler not compiled, ignore --profile");
            }
        }

        const auto start = std::chrono::steady_clock::now();
        pipeline.run();
        const auto end = std::chrono::steady_clock::now();

        const auto costMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        const double sec = static_cast<double>(costMs) / 1000.0;
        const size_t consumed = pipeline.getConsumedCount();
        const double throughput = (sec > 0.0) ? (static_cast<double>(consumed) / sec) : 0.0;

        LOG.info("========================================");
        LOG.info("Pipeline done in %lld ms", static_cast<long long>(costMs));
        LOG.info("Consumed: %zu, written: %zu, throughput: %.2f packets/s",
                 consumed,
                 consumer->getWrittenCount(),
                 throughput);
        LOG.info("========================================");

        if (enableProfiling)
        {
            if constexpr (GryFlux::Profiling::kBuildProfiling)
            {
                pipeline.printProfilingStats();
                const std::string timelinePath = "graph_timeline.json";
                pipeline.dumpProfilingTimeline(timelinePath);
                LOG.info("Graph timeline dumped to %s", timelinePath.c_str());
            }
        }
    }
    catch (const std::exception &e)
    {
        LOG.error("Fatal error: %s", e.what());
        return -1;
    }

    return 0;
}
