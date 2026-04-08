#include <iostream>
#include <string>
#include <memory>
#include <vector>

// GryFlux 框架核心
#include "framework/resource_pool.h"
#include "framework/graph_template.h"
#include "framework/template_builder.h"
#include "framework/async_pipeline.h"
#include "framework/profiler/profiling_build_config.h"

// 业务上下文与数据包
#include "context/infercontext.h"      
#include "context/reid_context.h"     
#include "packet/track_data_packet.h"

// 业务节点 (Nodes)
#include "nodes/preprocess/preprocess.h"
#include "nodes/infer/infer.h"
#include "nodes/postprocess/postprocess.h"
#include "nodes/reid_preprocess/reid_preprocess.h"
#include "nodes/reid_infer/reid_infer.h"

// 数据源与消费者
#include "source/image_data_source.h"
#include "consumer/result_consumer.h"

class TrackOutputNode : public GryFlux::NodeBase {
public:
    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override {}
};

void print_help() {
    std::cout << "Usage: deepsort_track_orin [OPTIONS]\n"
              << "Options:\n"
              << "  -h, --help            显示帮助信息\n"
              << "  -i, --input  <path>   输入视频流路径 (必填)\n"
              << "  -o, --output <path>   输出视频保存路径 (必填)\n"
              << "  -y, --yolox  <path>   YOLOX TensorRT engine 路径 (必填)\n"
              << "  -r, --reid   <path>   ReID TensorRT engine 路径 (必填)\n"
              << "  -t, --threads <num>   CPU 工作线程数 (默认: 12)\n"
              << "  -p, --packets <num>   最大活跃包数量 (默认: 8)\n"
              << "Example:\n"
              << "  ./deepsort_track_orin -i test.mp4 -o out.mp4 -y yolox.engine -r osnet.engine\n";
}

int main(int argc, char** argv) {
    std::string video_in = "";
    std::string video_out = "";
    std::string yolox_om = "";
    std::string reid_om = "";
    size_t threads = 12;
    size_t max_packets = 8;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help();
            return 0;
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            video_in = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            video_out = argv[++i];
        } else if ((arg == "-y" || arg == "--yolox") && i + 1 < argc) {
            yolox_om = argv[++i];
        } else if ((arg == "-r" || arg == "--reid") && i + 1 < argc) {
            reid_om = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            threads = std::stoul(argv[++i]);
        } else if ((arg == "-p" || arg == "--packets") && i + 1 < argc) {
            max_packets = std::stoul(argv[++i]);
        }
    }

    if (video_in.empty() || video_out.empty() || yolox_om.empty() || reid_om.empty()) {
        std::cerr << "错误: 缺少必要的参数！\n";
        print_help();
        return -1;
    }

    try {
        auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
        
        resourcePool->registerResourceType("npu", {
            std::make_shared<InferContext>(yolox_om, 0),
            std::make_shared<InferContext>(yolox_om, 0)
        });

        resourcePool->registerResourceType("reid_npu", {
            std::make_shared<ReidContext>(reid_om, 0),
            std::make_shared<ReidContext>(reid_om, 0)
        });

        auto graphTemplate = GryFlux::GraphTemplate::buildOnce([&](GryFlux::TemplateBuilder *builder) {
            builder->setInputNode<PreprocessNode>("preprocess", 640, 640);
            builder->addTask<InferNode>("yolox_infer", "npu", {"preprocess"});
            builder->addTask<PostprocessNode>("postprocess", "", {"yolox_infer"}, 640, 640, 0.3f, 0.45f);
            builder->addTask<ReidPreprocessNode>("reid_prep", "", {"postprocess"}, 128, 256);
            builder->addTask<ReidInferNode>("reid_infer", "reid_npu", {"reid_prep"}, 512);
            builder->setOutputNode<TrackOutputNode>("output", {"reid_infer"});
        });

        auto source = std::make_shared<ImageDataSource>(video_in);
        auto consumer = std::make_shared<ResultConsumer>(video_out, source->getFps(), source->getWidth(), source->getHeight());

        GryFlux::AsyncPipeline pipeline(source, graphTemplate, resourcePool, consumer, threads, max_packets);

        if constexpr (GryFlux::Profiling::kBuildProfiling) pipeline.setProfilingEnabled(true);

        std::cout << "GryFlux Orin TensorRT DeepSORT Pipeline Running..." << std::endl;
        pipeline.run(); 

        if constexpr (GryFlux::Profiling::kBuildProfiling) {
            pipeline.printProfilingStats();
            pipeline.dumpProfilingTimeline("graph_timeline.json");
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
