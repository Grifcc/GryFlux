#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "consumer/result_consumer.h"
#include "context/resnet_npu_context.h"
#include "nodes/resnet_nodes.h"
#include "source/image_dir_source.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct AppConfig
{
    std::string modelPath;
    std::string datasetDir;
    std::string synsetPath;
    std::string outputDir = "./outputs";
    bool enableProfiling = false;
};

void printHelp()
{
    std::cout << "Usage: resnet <model_path> <dataset_dir> <synset_path> [output_dir] [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --profile            Enable GryFlux profiling\n";
    std::cout << "  --help,-h            Show help and exit\n";
}

void initLogger()
{
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("resnet");
}

AppConfig parseArgs(int argc, char **argv)
{
    if (argc < 4)
    {
        throw std::runtime_error("Not enough arguments");
    }

    AppConfig config;
    config.modelPath = argv[1];
    config.datasetDir = argv[2];
    config.synsetPath = argv[3];

    int argIndex = 4;
    if (argIndex < argc && std::strncmp(argv[argIndex], "--", 2) != 0)
    {
        config.outputDir = argv[argIndex];
        ++argIndex;
    }

    while (argIndex < argc)
    {
        const char *arg = argv[argIndex];
        if (!std::strcmp(arg, "--help") || !std::strcmp(arg, "-h"))
        {
            printHelp();
            std::exit(0);
        }
        if (!std::strcmp(arg, "--profile"))
        {
            config.enableProfiling = true;
            ++argIndex;
            continue;
        }

        throw std::runtime_error(std::string("Unsupported option: ") + arg);
    }

    return config;
}

std::vector<std::string> loadLabels(const std::string &synsetPath)
{
    std::ifstream file(synsetPath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open synset file: " + synsetPath);
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line))
    {
        const std::size_t firstSpace = line.find(' ');
        if (firstSpace != std::string::npos && firstSpace + 1 < line.size())
        {
            labels.push_back(line.substr(firstSpace + 1));
        }
        else
        {
            labels.push_back(line);
        }
    }
    return labels;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc >= 2 && (!std::strcmp(argv[1], "--help") || !std::strcmp(argv[1], "-h")))
    {
        printHelp();
        return 0;
    }

    initLogger();

    constexpr int kModelWidth = 224;
    constexpr int kModelHeight = 224;
    constexpr std::size_t kTopK = 5;
    constexpr int kNpuInstances = 3;
    constexpr std::size_t kThreadPoolSize = 8;
    constexpr std::size_t kMaxActivePackets = 8;

    try
    {
        const AppConfig config = parseArgs(argc, argv);
        const auto classLabels = loadLabels(config.synsetPath);
        if (classLabels.empty())
        {
            LOG.warning("Synset file is empty, fallback labels will use class_<id>");
        }

        LOG.info("========================================");
        LOG.info("GryFlux ResNet Pipeline");
        LOG.info("Model : %s", config.modelPath.c_str());
        LOG.info("Input : %s", config.datasetDir.c_str());
        LOG.info("Synset: %s", config.synsetPath.c_str());
        LOG.info("Output: %s", config.outputDir.c_str());
        LOG.info("========================================");

        auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
        {
            std::vector<std::shared_ptr<GryFlux::Context>> npuContexts;
            npuContexts.reserve(static_cast<std::size_t>(kNpuInstances));
            for (int i = 0; i < kNpuInstances; ++i)
            {
                npuContexts.push_back(std::make_shared<ResnetNpuContext>(
                    i,
                    config.modelPath,
                    kModelWidth,
                    kModelHeight));
            }
            resourcePool->registerResourceType("npu", std::move(npuContexts));
        }

        auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
            [&](GryFlux::TemplateBuilder *builder)
            {
                builder->setInputNode<ResnetNodes::InputNode>("input");
                builder->addTask<ResnetNodes::PreprocessNode>("preprocess", "", {"input"}, kModelWidth, kModelHeight);
                builder->addTask<ResnetNodes::InferenceNode>("inference", "npu", {"preprocess"});
                builder->addTask<ResnetNodes::PostprocessNode>(
                    "postprocess",
                    "",
                    {"inference"},
                    classLabels,
                    kTopK);
                builder->setOutputNode<ResnetNodes::OutputNode>("output", {"postprocess"});
            });

        auto source = std::make_shared<ResnetImageDirSource>(config.datasetDir);
        auto consumer = std::make_shared<ResnetResultConsumer>(config.outputDir);

        GryFlux::AsyncPipeline pipeline(
            source,
            graphTemplate,
            resourcePool,
            consumer,
            kThreadPoolSize,
            kMaxActivePackets);

        if (config.enableProfiling)
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
        const double seconds = static_cast<double>(costMs) / 1000.0;
        const std::size_t consumed = consumer->getConsumedCount();
        const std::size_t written = consumer->getWrittenCount();
        const double throughput = (seconds > 0.0) ? (static_cast<double>(consumed) / seconds) : 0.0;

        LOG.info("========================================");
        LOG.info("Pipeline done in %lld ms", static_cast<long long>(costMs));
        LOG.info("Consumed: %zu, written: %zu, throughput: %.2f packets/s",
                 consumed,
                 written,
                 throughput);
        LOG.info("========================================");

        if (config.enableProfiling)
        {
            if constexpr (GryFlux::Profiling::kBuildProfiling)
            {
                pipeline.printProfilingStats();
                const std::string timelinePath = "resnet_graph_timeline.json";
                pipeline.dumpProfilingTimeline(timelinePath);
                LOG.info("Graph timeline dumped to %s", timelinePath.c_str());
            }
        }
    }
    catch (const std::exception &e)
    {
        LOG.error("Fatal error: %s", e.what());
        printHelp();
        return -1;
    }

    return 0;
}
