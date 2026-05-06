#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <system_error>

#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"

#include "packet/ZeroDce_Packet.h"
#include "source/ZeroDceDataSource.h" 
#include "consumer/ResultConsumer/ZeroDceResultConsumer.h"
#include "consumer/DiskWriter/AsyncDiskWriter.h"

#include "context/AtlasContext.h"
#include "nodes/Input/InputNode.h"
#include "nodes/Output/OutputNode.h"
#include "nodes/Preprocess/PreprocessNode.h"
#include "nodes/Infer/InferNode.h"
#include "nodes/Postprocess/PostprocessNode.h"

#include <opencv2/opencv.hpp>

namespace {

std::filesystem::path ResolveExecutableDir(const char* argv0) {
    std::error_code ec;
    std::filesystem::path exe_path = (argv0 == nullptr) ? std::filesystem::path() : std::filesystem::path(argv0);

    if (!exe_path.is_absolute()) {
        exe_path = std::filesystem::absolute(exe_path, ec);
        ec.clear();
    }

    const std::filesystem::path normalized_path = std::filesystem::weakly_canonical(exe_path, ec);
    if (!ec && !normalized_path.empty()) {
        exe_path = normalized_path;
    }

    if (exe_path.has_parent_path()) {
        return exe_path.parent_path();
    }

    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec) {
        return cwd;
    }

    return ".";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "用法: zero_dce_app <om_model_path> <input_dir> <output_dir>" << std::endl;
        return 1;
    }

    try {
        const std::string omModelPath = argv[1];
        const std::string inputDir = argv[2];
        const std::string outputDir = argv[3];

        cv::setNumThreads(0);

        {
            auto source = std::make_shared<ZeroDceDataSource>(inputDir);
            if (source->GetTotalFrames() == 0) {
                std::cerr << "未找到可处理的输入图片！" << std::endl;
                return 1;
            }

            std::cout << "[INFO] 开始初始化 ZeroDCE 流水线。" << std::endl;

            auto consumer = std::make_shared<ZeroDceResultConsumer>(source->GetTotalFrames());

            auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
            std::vector<std::shared_ptr<GryFlux::Context>> atlas_contexts;

            auto device0_contexts = CreateAtlasContexts(omModelPath, 0, 1);
            auto device1_contexts = CreateAtlasContexts(omModelPath, 1, 1);
            atlas_contexts.insert(atlas_contexts.end(),
                                  device0_contexts.begin(),
                                  device0_contexts.end());
            atlas_contexts.insert(atlas_contexts.end(),
                                  device1_contexts.begin(),
                                  device1_contexts.end());

            resourcePool->registerResourceType("atlas_npu", std::move(atlas_contexts));

            auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
                [](GryFlux::TemplateBuilder *builder) {
                    builder->setInputNode<InputNode>("input");
                    builder->addTask<PreprocessNode>("preprocess", "", {"input"});
                    builder->addTask<InferNode>("inference", "atlas_npu", {"preprocess"});
                    builder->addTask<PostprocessNode>("postprocess", "", {"inference"});
                    builder->setOutputNode<OutputNode>("output", {"postprocess"});
                }
            );

            constexpr size_t kThreadPoolSize = 8;
            constexpr size_t kMaxActivePackets = 16;
            GryFlux::AsyncPipeline pipeline(
                source,
                graphTemplate,
                resourcePool,
                consumer,
                kThreadPoolSize,
                kMaxActivePackets
            );

            AsyncDiskWriter::GetInstance().Start(outputDir);

            if constexpr (GryFlux::Profiling::kBuildProfiling) {
                pipeline.setProfilingEnabled(true);
            }

            std::cout << "[INFO] 开始处理图片..." << std::endl;

            pipeline.run();

            if constexpr (GryFlux::Profiling::kBuildProfiling) {
                const std::filesystem::path timeline_path =
                    ResolveExecutableDir(argv[0]) / "graph_timeline_zero_dce.json";
                pipeline.dumpProfilingTimeline(timeline_path.string());
                std::cout << "[INFO] Profiling 时间线已导出: "
                          << timeline_path.string() << std::endl;
            }

            consumer->printMetrics();
            AsyncDiskWriter::GetInstance().Stop();
        }

        std::cout << "[INFO] ACL 资源已释放，程序结束。" << std::endl;
        return 0;
    } catch (const std::exception& exception) {
        AsyncDiskWriter::GetInstance().Stop();
        std::cerr << "[ERROR] " << exception.what() << std::endl;
        return 1;
    }
}
