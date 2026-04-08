#include <iostream>
#include <string>
#include <memory>
#include <vector>

// GryFlux 框架核心组件
#include "framework/resource_pool.h"
#include "framework/graph_template.h"
#include "framework/template_builder.h"
#include "framework/async_pipeline.h"
#include "framework/profiler/profiling_build_config.h"

// 业务组件
#include "source/image_data_source.h"
#include "consumer/result_consumer.h"
#include "context/infercontext.h"
#include "nodes/preprocess/preprocess.h"
#include "nodes/infer/infer.h"
#include "nodes/postprocess/postprocess.h"

void print_help() {
    std::cout << "Usage: yolox_detection_orin [OPTIONS]\n"
              << "Options:\n"
              << "  -h, --help            显示帮助信息\n"
              << "  -i, --input  <path>   输入视频流路径 (必填)\n"
              << "  -o, --output <path>   输出视频路径 (必填)\n"
              << "  -m, --model  <path>   TensorRT engine 路径 (必填)\n"
              << "  -w, --width  <num>    模型输入宽 (默认: 640)\n"
              << "  -H, --height <num>    模型输入高 (默认: 640)\n"
              << "  -c, --conf   <float>  置信度阈值 (默认: 0.3)\n"
              << "  -n, --nms    <float>  NMS 阈值 (默认: 0.45)\n";
}

int main(int argc, char** argv) {
    std::string video_in = "";
    std::string video_out = "";
    std::string om_path = "";
    int model_w = 640;
    int model_h = 640;
    float conf_thresh = 0.3f;
    float nms_thresh = 0.45f;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(); return 0;
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            video_in = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            video_out = argv[++i];
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            om_path = argv[++i];
        } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            model_w = std::stoi(argv[++i]);
        } else if ((arg == "-H" || arg == "--height") && i + 1 < argc) {
            model_h = std::stoi(argv[++i]);
        } else if ((arg == "-c" || arg == "--conf") && i + 1 < argc) {
            conf_thresh = std::stof(argv[++i]);
        } else if ((arg == "-n" || arg == "--nms") && i + 1 < argc) {
            nms_thresh = std::stof(argv[++i]);
        }
    }

    if (video_in.empty() || video_out.empty() || om_path.empty()) {
        std::cerr << "错误: 缺少必要的参数！\n";
        print_help(); return -1;
    }

    try {
        auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
        
        resourcePool->registerResourceType("npu", {
            std::make_shared<InferContext>(om_path, 0),
            std::make_shared<InferContext>(om_path, 0)
        });

        auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
            [&](GryFlux::TemplateBuilder *builder) {
                builder->setInputNode<PreprocessNode>("preprocess", model_w, model_h);
                builder->addTask<InferNode>("infer", "npu", {"preprocess"});
                builder->setOutputNode<PostprocessNode>("postprocess", {"infer"}, model_w, model_h, conf_thresh, nms_thresh);
            }
        );

        auto source = std::make_shared<ImageDataSource>(video_in);
        auto consumer = std::make_shared<ResultConsumer>(video_out, source->getFps(), source->getWidth(), source->getHeight());

        size_t thread_pool_size = 12;     
        size_t max_active_packets = 8;   

        GryFlux::AsyncPipeline pipeline(source, graphTemplate, resourcePool, consumer, thread_pool_size, max_active_packets);

        if constexpr (GryFlux::Profiling::kBuildProfiling) pipeline.setProfilingEnabled(true);

        std::cout << "========== 启动 GryFlux Orin TensorRT 异步管道 ==========" << std::endl;
        pipeline.run(); 

        if constexpr (GryFlux::Profiling::kBuildProfiling) {
            pipeline.printProfilingStats();
            pipeline.dumpProfilingTimeline("graph_timeline.json");
        }
    } catch (const std::exception& e) {
        std::cerr << "程序发生异常: " << e.what() << std::endl;
    }

    return 0;
}
