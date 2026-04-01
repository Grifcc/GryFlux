#include <iostream>
#include <string>
#include <chrono>
#include "acl/acl.h"
#include "framework/async_pipeline.h"
#include "framework/template_builder.h"
#include "packet/ZeroDce_Packet.h"
#include "source/ZeroDceDataSource.h" 
#include "consumer/ResultConsumer/ZeroDceResultConsumer.h"
#include "consumer/DiskWriter/AsyncDiskWriter.h"

#include "context/AtlasContext.h"
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

    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "ACL 初始化失败！" << std::endl;
        return 1;
    }

    std::cout << "[INFO] --- 开始配置 Zero-DCE 图像增强异步流水线 ---" << std::endl;

    AsyncDiskWriter::GetInstance().Start(outputDir);

    auto source = std::make_shared<ZeroDceDataSource>(inputDir);
    auto consumer = std::make_shared<ZeroDceResultConsumer>(source->GetTotalFrames());

    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
    std::vector<std::shared_ptr<GryFlux::Context>> atlas_contexts;

    atlas_contexts.push_back(std::make_shared<AtlasContext>(0, omModelPath));
    atlas_contexts.push_back(std::make_shared<AtlasContext>(1, omModelPath)); 
    
    resourcePool->registerResourceType("atlas_npu", std::move(atlas_contexts));

    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder *builder) {
            builder->setInputNode<PreprocessNode>("preprocess");
            builder->addTask<InferNode>("inference", "atlas_npu", {"preprocess"});
            builder->setOutputNode<PostprocessNode>("postprocess", {"inference"});
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

    std::cout << "[INFO] --- 引擎点火，开始极速图像增强 ---" << std::endl;
    
    auto finish_signal = consumer->get_future();

    std::thread pipeline_thread([&pipeline]() {
        pipeline.run();
    });

    finish_signal.get();

    if (pipeline_thread.joinable()) {
        pipeline_thread.join();
    }

    consumer->printMetrics();

    AsyncDiskWriter::GetInstance().Stop();

    aclFinalize();
    std::cout << "[INFO] ACL 硬件资源及 I/O 线程已完全释放，程序优雅退出。" << std::endl;
    
    return 0;
}