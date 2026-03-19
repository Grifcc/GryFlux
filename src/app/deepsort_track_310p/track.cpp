#include <iostream>
#include <string>
#include <memory>

// ACL 核心
#include "acl/acl.h"

// GryFlux 框架核心
#include "framework/resource_pool.h"
#include "framework/graph_template.h"
#include "framework/template_builder.h"
#include "framework/async_pipeline.h"
#include "framework/profiler/profiling_build_config.h"

// 业务上下文与数据包
#include "context/infercontext.h"      // YOLOX 硬件上下文
#include "context/reid_context.h"     // ReID 硬件上下文
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

// 在 include 区域下方添加这个简单的透传节点
class TrackOutputNode : public GryFlux::NodeBase {
public:
    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override {
        // 什么都不做，纯粹作为流水线终点，触发 ResultConsumer
    }
};

int main() {
    // 0. ACL 全局初始化
    if (aclInit(nullptr) != ACL_SUCCESS) {
        return -1;
    }

    // --- 1. 配置参数 (建议使用绝对路径) ---
    std::string video_in = "/root/workspace/ma/yolox_deepsort_ascent/data/videos/test.mp4";
    std::string video_out = "track_result.mp4";
    std::string yolox_om = "/root/workspace/ma/GryFlux/models/yolox_s_9.om";
    std::string reid_om = "/root/workspace/ma/GryFlux/models/osnet_x0_25.om";

    try {
        // --- 2. 硬件资源池注册 (双 NPU 负载均衡) ---
        auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
        
        // 注册检测资源：两块 NPU 同时服务于 YOLOX
        resourcePool->registerResourceType("npu", {
            std::make_shared<InferContext>(yolox_om, 0),
            std::make_shared<InferContext>(yolox_om, 1)
        });

        // 注册特征资源：两块 NPU 同时服务于 ReID
        resourcePool->registerResourceType("reid_npu", {
            std::make_shared<ReidContext>(reid_om, 0),
            std::make_shared<ReidContext>(reid_om, 1)
        });

        // --- 3. 构建 5 级 DAG 拓扑图 ---
        auto graphTemplate = GryFlux::GraphTemplate::buildOnce([&](GryFlux::TemplateBuilder *builder) {
            // 第一级：全图预处理 (CPU)
            builder->setInputNode<PreprocessNode>("preprocess", 640, 640);

            // 第二级：YOLOX 推理 (使用 npu 资源)
            builder->addTask<InferNode>("yolox_infer", "npu", {"preprocess"});

            // 第三级：YOLOX 后处理 (CPU, 拿到检测框)
            builder->addTask<PostprocessNode>("postprocess", "", {"yolox_infer"}, 640, 640, 0.3f, 0.45f);

            // 第四级：ReID 局部预处理 (CPU, 抠图并 Resize)
            builder->addTask<ReidPreprocessNode>("reid_prep", "", {"postprocess"}, 128, 256);

            // 第五级：ReID 特征提取 (使用 reid_npu 资源)
            // 1. 使用 addTask 绑定 NPU 资源
            builder->addTask<ReidInferNode>("reid_infer", "reid_npu", {"reid_prep"}, 512);

            // 2. 使用我们刚写的透传节点作为终点
            builder->setOutputNode<TrackOutputNode>("output", {"reid_infer"});
        });

        // --- 4. 初始化 IO 组件 ---
        auto source = std::make_shared<ImageDataSource>(video_in);
        auto consumer = std::make_shared<ResultConsumer>(video_out, source->getFps(), source->getWidth(), source->getHeight());

        // --- 5. 启动异步流水线 ---
        size_t threads = 12; // i5-13500H 有 16 线程，分配 12 个给工作池
        size_t max_packets = 8; // 允许 8 帧同时在管道中奔跑

        GryFlux::AsyncPipeline pipeline(source, graphTemplate, resourcePool, consumer, threads, max_packets);

        // 开启分析
        if constexpr (GryFlux::Profiling::kBuildProfiling) {
            pipeline.setProfilingEnabled(true);
        }

        std::cout << "GryFlux Dual-NPU DeepSORT Pipeline Running..." << std::endl;
        pipeline.run(); // 阻塞运行直至结束

        // --- 6. 导出性能数据 ---
        if constexpr (GryFlux::Profiling::kBuildProfiling) {
            pipeline.printProfilingStats();
            pipeline.dumpProfilingTimeline("graph_timeline.json");
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    aclFinalize();
    return 0;
}