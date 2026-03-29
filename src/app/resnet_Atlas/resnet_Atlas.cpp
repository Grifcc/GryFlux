// src/resnet_Atlas/main_gryflux.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <chrono>
#include <thread>

#include "acl/acl.h"
#include "framework/async_pipeline.h"
#include "framework/template_builder.h"

// 引入我们刚刚写的四大金刚
#include "packet/resnet_packet.h"
#include "source/resnet_data_source.h"
#include "consumer/resnet_result_consumer.h"
#include "context/atlas_context.h"
#include "nodes/Preprocess/PreprocessNode.h"
#include "nodes/Infer/InferNode.h"
#include "nodes/Postprocess/PostprocessNode.h"

#include <opencv2/opencv.hpp>

// 照搬你原本加载标签的辅助函数
std::map<std::string, int> LoadGroundTruth(const std::string& gt_file_path) {
    std::map<std::string, int> gt_map;
    std::ifstream ifs(gt_file_path);
    if (!ifs.is_open()) {
        std::cerr << "错误: 无法打开真实标签文件: " << gt_file_path << std::endl;
        return gt_map;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        size_t split_pos = line.find(' ');
        if (split_pos == std::string::npos) continue;
        std::string filename = line.substr(0, split_pos);
        int label_index = std::stoi(line.substr(split_pos + 1));
        gt_map[filename] = label_index;
    }
    return gt_map;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "用法: ./classification_app_gryflux <om_model_path> <dataset_dir> <gt_file_path>" << std::endl;
        return 1;
    }
    std::string omModelPath = argv[1];
    std::string datasetDir = argv[2];
    std::string gtFilePath = argv[3];

    // 🌟 解决卡死的关键：强制 OpenCV 变为单线程，彻底杜绝底层线程爆炸！
    cv::setNumThreads(0);

    // 1. 全局初始化 ACL
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "ACL 初始化失败！" << std::endl;
        return 1;
    }

    std::cout << "[INFO] --- 开始配置 GryFlux 异步流水线 ---" << std::endl;

    // 2. 加载数据集标签，初始化 Source 和 Consumer
    auto gt_map = LoadGroundTruth(gtFilePath);
    if (gt_map.empty()) return 1;
    
    auto source = std::make_shared<ResNetDataSource>(datasetDir, gt_map);
    auto consumer = std::make_shared<ResNetResultConsumer>(gt_map.size());

    // 3. 注册 Atlas 硬件资源池
    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
    std::vector<std::shared_ptr<GryFlux::Context>> atlas_contexts;
    
    // 🌟 性能翻倍的关键：同时注册两块 NPU（Device 0 和 Device 1）
    // 框架会自动在这两张卡之间做负载均衡，轮询分配推理任务
    atlas_contexts.push_back(std::make_shared<AtlasContext>(0, omModelPath));
    atlas_contexts.push_back(std::make_shared<AtlasContext>(1, omModelPath)); 
    
    resourcePool->registerResourceType("atlas_npu", std::move(atlas_contexts));

    // 4. 构建 DAG 拓扑图
    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder *builder) {
            // CPU 预处理节点
            builder->setInputNode<PreprocessNode>("preprocess");
            
            // NPU 推理节点，绑定 "atlas_npu" 资源，依赖 "preprocess" 完成
            builder->addTask<ResNetInferNode>("inference", "atlas_npu", {"preprocess"});
            
            // CPU 后处理节点，依赖 "inference" 完成
            builder->setOutputNode<PostprocessNode>("postprocess", {"inference"});
        }
    );

    // 5. 配置并启动异步管道
    constexpr size_t kThreadPoolSize = 8;      // CPU 线程池大小（负责预处理和后处理）
    constexpr size_t kMaxActivePackets = 16;   // 流水线最大同时存在的包数（背压控制）

    GryFlux::AsyncPipeline pipeline(
        source, 
        graphTemplate, 
        resourcePool, 
        consumer, 
        kThreadPoolSize, 
        kMaxActivePackets
    );

    std::cout << "[INFO] --- 引擎点火，开始异步评估 ---" << std::endl;
    
    // 🌟 获取 Consumer 的信号接收器
    auto finish_signal = consumer->get_future();

    // 🌟 把 pipeline 扔进后台独立线程去跑，防止它锁死主程序
    std::thread pipeline_thread([&pipeline]() {
        pipeline.run();
    });

    // 🌟 主程序在这里乖乖等待，直到收到 Consumer 的 "处理完成" 信号
    finish_signal.get();

    if (pipeline_thread.joinable()) {
        pipeline_thread.join();
    }

    // 收到信号后，开始优雅结算
    consumer->printMetrics();

    // 🌟 正规军的做法：调用华为底层接口，释放所有 NPU 显存和资源
    aclFinalize();
    std::cout << "[INFO] ACL 底层硬件资源已完全释放，程序优雅退出。" << std::endl;
    
    return 0;
}
