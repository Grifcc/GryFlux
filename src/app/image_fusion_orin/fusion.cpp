// GryFlux Framework 依赖
#include "framework/resource_pool.h"
#include "framework/graph_template.h"
#include "framework/template_builder.h"
#include "framework/async_pipeline.h"
#include "framework/profiler/profiling_build_config.h"
#include "utils/logger.h"

// 业务节点依赖
#include "context/infercontext.h"
#include "packet/fusion_data_packet.h"
#include "source/fusion_data_source.h"
#include "consumer/result_consumer.h"
#include "nodes/preprocess/preprocess.h"
#include "nodes/infer/infer.h"
#include "nodes/postprocess/postprocess.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

namespace {

bool probeFusionInputSize(const std::string& visDir, int& width, int& height) {
    namespace fs = std::filesystem;

    if (!fs::exists(visDir)) {
        return false;
    }

    for (const auto& entry : fs::directory_iterator(visDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            continue;
        }

        width = image.cols;
        height = image.rows;
        return true;
    }

    return false;
}

}

void print_help() {
    std::cout << "Usage: image_fusion_orin [OPTIONS]\n"
              << "Options:\n"
              << "  -h, --help            显示帮助信息\n"
              << "  -v, --vis   <dir>     可见光图像目录 (必填)\n"
              << "  -i, --ir    <dir>     红外图像目录 (必填)\n"
              << "  -s, --save  <dir>     融合结果保存目录 (必填)\n"
              << "  -m, --model <path>    融合 TensorRT engine 路径 (必填)\n"
              << "  -t, --threads <num>   工作线程数 (默认: 8)\n"
              << "  -n, --npu   <num>     推理实例数 (默认: 2)\n";
}

int main(int argc, char **argv) {
    std::string visDir = "";
    std::string irDir = "";
    std::string saveDir = "";
    std::string modelPath = "";
    size_t kThreadPoolSize = 8;
    size_t kNpuInstances = 2;
    constexpr size_t kMaxActivePackets = 16; 

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(); return 0;
        } else if ((arg == "-v" || arg == "--vis") && i + 1 < argc) {
            visDir = argv[++i];
        } else if ((arg == "-i" || arg == "--ir") && i + 1 < argc) {
            irDir = argv[++i];
        } else if ((arg == "-s" || arg == "--save") && i + 1 < argc) {
            saveDir = argv[++i];
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            modelPath = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            kThreadPoolSize = std::stoul(argv[++i]);
        } else if ((arg == "-n" || arg == "--npu") && i + 1 < argc) {
            kNpuInstances = std::stoul(argv[++i]);
        }
    }

    if (visDir.empty() || irDir.empty() || saveDir.empty() || modelPath.empty()) {
        std::cerr << "错误: 缺少必要的参数！\n";
        print_help(); return -1;
    }

    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("ImageFusionOrin");

    int input_width = GetFusionModelWidth();
    int input_height = GetFusionModelHeight();
    if (probeFusionInputSize(visDir, input_width, input_height)) {
        SetFusionModelSize(input_width, input_height);
        LOG.info("Fusion input probe size: %dx%d", input_width, input_height);
    }

    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
    std::vector<std::shared_ptr<GryFlux::Context>> npuContexts;
    npuContexts.reserve(kNpuInstances);
    
    for (size_t i = 0; i < kNpuInstances; ++i) {
        auto ctx = std::make_shared<InferContext>();
        if (!ctx->Init(modelPath, 0)) {
            LOG.error("InferContext 初始化失败"); return -1;
        }
        npuContexts.push_back(ctx);
    }
    
    resourcePool->registerResourceType("image_fusion_npu", std::move(npuContexts));

    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder *builder) {
            builder->setInputNode<PreprocessNode>("preprocess");
            builder->addTask<InferNode>("infer", "image_fusion_npu", {"preprocess"});
            builder->setOutputNode<PostprocessNode>("postprocess", {"infer"});
        }
    );

    auto source = std::make_shared<FusionDataSource>(visDir, irDir);
    auto consumer = std::make_shared<FusionDataConsumer>(saveDir);

    GryFlux::AsyncPipeline pipeline(source, graphTemplate, resourcePool, consumer, kThreadPoolSize, kMaxActivePackets);

    if constexpr (GryFlux::Profiling::kBuildProfiling) pipeline.setProfilingEnabled(true);

    pipeline.run();

    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        pipeline.printProfilingStats();
        pipeline.dumpProfilingTimeline("fusion_timeline.json");
    }

    resourcePool.reset(); 
    return 0;
}
