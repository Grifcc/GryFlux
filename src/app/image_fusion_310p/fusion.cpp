#include "acl/acl.h"

// GryFlux Framework 依赖
#include "framework/resource_pool.h"
#include "framework/graph_template.h"
#include "framework/template_builder.h"
#include "framework/async_pipeline.h"
#include "framework/profiler/profiling_build_config.h"
#include "utils/logger.h"

// 业务节点依赖
#include "context/infercontext.h"
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

void print_help() {
    std::cout << "Usage: fusion_310p [OPTIONS]\n"
              << "Options:\n"
              << "  -h, --help            显示帮助信息\n"
              << "  -v, --vis   <dir>     可见光图像目录 (必填)\n"
              << "  -i, --ir    <dir>     红外图像目录 (必填)\n"
              << "  -s, --save  <dir>     融合结果保存目录 (必填)\n"
              << "  -m, --model <path>    融合模型 .om 路径 (必填)\n"
              << "  -t, --threads <num>   工作线程数 (默认: 8)\n"
              << "  -n, --npu   <num>     NPU 实例数 (默认: 2)\n";
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
    LOG.setAppName("ImageFusion310P");

    if (aclInit("") != ACL_ERROR_NONE) return -1;

    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
    std::vector<std::shared_ptr<GryFlux::Context>> npuContexts;
    npuContexts.reserve(kNpuInstances);
    
    for (size_t i = 0; i < kNpuInstances; ++i) {
        auto ctx = std::make_shared<InferContext>();
        // 将偶数实例绑到 device 0，奇数实例绑到 device 1，实现双卡并发
        if (!ctx->Init(modelPath, i % 2)) {
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
    aclFinalize();
    return 0;
}