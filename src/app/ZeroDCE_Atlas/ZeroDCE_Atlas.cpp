#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "用法: ./zero_dce_app <om_model_path> <input_dir> <output_dir>" << std::endl;
        return 1;
    }
    std::string omModelPath = argv[1];
    std::string inputDir = argv[2];
    std::string outputDir = argv[3];

    cv::setNumThreads(0);

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

    consumer->printMetrics();

    AsyncDiskWriter::GetInstance().Stop();

    return 0;
}
