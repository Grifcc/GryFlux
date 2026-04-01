#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <chrono>
#include <thread>

#include "acl/acl.h"
#include "framework/async_pipeline.h"
#include "framework/template_builder.h"

#include "packet/resnet_packet.h"
#include "source/resnet_data_source.h"
#include "consumer/resnet_result_consumer.h"
#include "context/atlas_context.h"
#include "nodes/Preprocess/PreprocessNode.h"
#include "nodes/Infer/InferNode.h"
#include "nodes/Postprocess/PostprocessNode.h"

#include <opencv2/opencv.hpp>

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


    cv::setNumThreads(0);

    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "ACL 初始化失败！" << std::endl;
        return 1;
    }

    std::cout << "[INFO] --- 开始配置 GryFlux 异步流水线 ---" << std::endl;

    auto gt_map = LoadGroundTruth(gtFilePath);
    if (gt_map.empty()) return 1;
    
    auto source = std::make_shared<ResNetDataSource>(datasetDir, gt_map);
    auto consumer = std::make_shared<ResNetResultConsumer>(gt_map.size());

    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
    std::vector<std::shared_ptr<GryFlux::Context>> atlas_contexts;
    
    atlas_contexts.push_back(std::make_shared<AtlasContext>(0, omModelPath));
    atlas_contexts.push_back(std::make_shared<AtlasContext>(1, omModelPath)); 
    
    resourcePool->registerResourceType("atlas_npu", std::move(atlas_contexts));


    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder *builder) {
            builder->setInputNode<PreprocessNode>("preprocess");
            
            builder->addTask<ResNetInferNode>("inference", "atlas_npu", {"preprocess"});
            
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

    std::cout << "[INFO] --- 引擎点火，开始异步评估 ---" << std::endl;
    

    auto finish_signal = consumer->get_future();

    std::thread pipeline_thread([&pipeline]() {
        pipeline.run();
    });

    finish_signal.get();

    if (pipeline_thread.joinable()) {
        pipeline_thread.join();
    }

    consumer->printMetrics();

    aclFinalize();
    std::cout << "[INFO] ACL 底层硬件资源已完全释放，程序优雅退出。" << std::endl;
    
    return 0;
}
