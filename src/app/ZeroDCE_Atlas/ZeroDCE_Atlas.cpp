// app/ZeroDCE_Atlas/source/main.cpp
#include <iostream>
#include <string>
#include <chrono>

#include "acl/acl.h"
#include "framework/async_pipeline.h"
#include "framework/template_builder.h"

// 引入 Zero-DCE 的专属组件
#include "packet/ZeroDce_Packet.h"
// 注意：你需要根据 ResNet 的逻辑，仿写这两个文件（负责读取目录图片和计数）
#include "source/ZeroDceDataSource.h" 
// 补上 ResultConsumer 和 DiskWriter 这两个子文件夹层级
#include "consumer/ResultConsumer/ZeroDceResultConsumer.h"
#include "consumer/DiskWriter/AsyncDiskWriter.h"

// NPU 上下文与核心节点
#include "context/AtlasContext.h"
#include "nodes/Preprocess/PreprocessNode.h"
#include "nodes/Infer/InferNode.h"
#include "nodes/Postprocess/PostprocessNode.h"

#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    // Zero-DCE 不需要 ground truth 标签，只需要模型、输入目录和输出目录
    if (argc != 4) {
        std::cout << "用法: ./zero_dce_app <om_model_path> <input_dir> <output_dir>" << std::endl;
        return 1;
    }
    std::string omModelPath = argv[1];
    std::string inputDir = argv[2];
    std::string outputDir = argv[3];

    // 🌟 解决卡死的关键：强制 OpenCV 变为单线程
    cv::setNumThreads(0);

    // 1. 全局初始化 ACL
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "ACL 初始化失败！" << std::endl;
        return 1;
    }

    std::cout << "[INFO] --- 开始配置 Zero-DCE 图像增强异步流水线 ---" << std::endl;

    // 🌟 核心新增：启动后台异步写盘队列，保护主干道吞吐量
    AsyncDiskWriter::GetInstance().Start(outputDir);

    // 2. 初始化 Source 和 Consumer
    // Source 负责遍历 inputDir 将图片塞入流水线，Consumer 负责统计完成的帧数
    auto source = std::make_shared<ZeroDceDataSource>(inputDir);
    auto consumer = std::make_shared<ZeroDceResultConsumer>(source->GetTotalFrames());

    // 3. 注册双 Atlas 310P1 硬件资源池
    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
    std::vector<std::shared_ptr<GryFlux::Context>> atlas_contexts;
    
    // 依然使用双卡并发调度
    atlas_contexts.push_back(std::make_shared<AtlasContext>(0, omModelPath));
    atlas_contexts.push_back(std::make_shared<AtlasContext>(1, omModelPath)); 
    
    resourcePool->registerResourceType("atlas_npu", std::move(atlas_contexts));

    // 4. 构建 DAG 拓扑图 (严格对应你刚才创建的三个 Node)
    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder *builder) {
            builder->setInputNode<PreprocessNode>("preprocess");
            builder->addTask<InferNode>("inference", "atlas_npu", {"preprocess"});
            builder->setOutputNode<PostprocessNode>("postprocess", {"inference"});
        }
    );

    // 5. 配置并启动异步管道
    constexpr size_t kThreadPoolSize = 8;      // CPU 线程池处理图片解码和通道转换
    constexpr size_t kMaxActivePackets = 16;   // 背压控制：内存池中同时存在的最大 Packet 数

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

    // 🌟 主程序阻塞，等待所有图片走完流水线
    finish_signal.get();

    if (pipeline_thread.joinable()) {
        pipeline_thread.join();
    }

    // 6. 优雅结算与资源释放
    consumer->printMetrics();

    // 🌟 核心新增：停止异步写盘队列，确保最后几张图安全落盘
    AsyncDiskWriter::GetInstance().Stop();

    aclFinalize();
    std::cout << "[INFO] ACL 硬件资源及 I/O 线程已完全释放，程序优雅退出。" << std::endl;
    
    return 0;
}