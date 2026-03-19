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

int main(int argc, char **argv) {
    // ==========================================
    // 0. 初始化日志记录器
    // ==========================================
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("ImageFusion310P");

    LOG.info("========================================");
    LOG.info("  Ascend 310P3 图像融合高并发流水线启动  ");
    LOG.info("========================================");

    // ==========================================
    // 1. 初始化 Ascend 系统环境
    // ==========================================
    const char* aclConfigPath = "";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        LOG.error("ACL 初始化失败, 错误码: %d", ret);
        return -1;
    }

    // ==========================================
    // 2. 参数配置
    // ==========================================
    std::string modelPath = "/root/workspace/ma/GryFlux/models/fusionnetv2.om";
    std::string visDir = "/root/workspace/ma/GryFlux/data/test_imgs/visible";
    std::string irDir = "/root/workspace/ma/GryFlux/data/test_imgs/infrared";
    std::string saveDir = "/root/workspace/ma/GryFlux/data/test_imgs/fusion";
    int deviceId = 0;

    constexpr size_t kThreadPoolSize = 8;     // GryFlux 框架执行任务的 CPU 线程数
    constexpr size_t kMaxActivePackets = 16;  // 允许在流水线中同时排队流转的最大帧数
    constexpr size_t kNpuInstances = 4;       // NPU 允许并发执行的模型实例数

    // ==========================================
    // 3. 构建硬件资源池 (Resource Pool)
    // ==========================================
    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
    std::vector<std::shared_ptr<GryFlux::Context>> npuContexts;
    npuContexts.reserve(kNpuInstances);
    
    for (size_t i = 0; i < kNpuInstances; ++i) {
        auto ctx = std::make_shared<InferContext>();
        if (!ctx->Init(modelPath, deviceId)) {
            LOG.error("InferContext 初始化失败: 实例 %zu", i);
            return -1;
        }
        npuContexts.push_back(ctx);
    }
    // 将分配好的 4 个 Ascend 资源注册给框架统一调度
    resourcePool->registerResourceType("image_fusion_npu", std::move(npuContexts));
    LOG.info("成功注册 %zu 个 NPU 资源实例", kNpuInstances);

    // ==========================================
    // 4. 构建数据流计算图 (Graph Template)
    // ==========================================
    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder *builder) {
            // 第一层：输入预处理节点 (不需要硬件资源申请)
            builder->setInputNode<PreprocessNode>("preprocess");
            
            // 第二层：推理节点 (依赖 preprocess 的输出，且必须向 resourcePool 申请 "image_fusion_npu" 资源)
            builder->addTask<InferNode>("infer", "image_fusion_npu", {"preprocess"});
            
            // 第三层：输出后处理节点 (依赖 infer 的输出)
            builder->setOutputNode<PostprocessNode>("postprocess", {"infer"});
        }
    );

    // ==========================================
    // 5. 实例化数据源和消费端
    // ==========================================
    auto source = std::make_shared<FusionDataSource>(visDir, irDir);
    auto consumer = std::make_shared<FusionDataConsumer>(saveDir);

    // ==========================================
    // 6. 启动异步并发流水线
    // ==========================================
    GryFlux::AsyncPipeline pipeline(
        source,
        graphTemplate,
        resourcePool,
        consumer,
        kThreadPoolSize,
        kMaxActivePackets
    );

    // 启用框架自带的 Profiler (如果在 CMake 中开启了宏)
    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        pipeline.setProfilingEnabled(true);
    }

    LOG.info("开始执行推理...");
    auto startTime = std::chrono::steady_clock::now();
    
    // 这一步会阻塞，直到 Data Source 返回 false，并且所有任务处理落盘完毕
    pipeline.run();

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    LOG.info("流水线执行完毕，总耗时: %lld ms", duration.count());

    if constexpr (GryFlux::Profiling::kBuildProfiling) {
        pipeline.printProfilingStats();
        pipeline.dumpProfilingTimeline("fusion_timeline.json");
    }

    // ==========================================
    // 7. 资源清理与退出 (非常重要)
    // ==========================================
    // 在调用 aclFinalize 之前，必须先将所有的 Device 内存和模型资源释放！
    // 因为它们被包装在 resourcePool 里的 std::shared_ptr 中，如果不主动清空，
    // main 函数结束时 shared_ptr 的析构函数会试图调用 aclrtFree，但那时 ACL 已经 Finalize 了，会导致进程崩溃。
    resourcePool.reset(); 

    aclFinalize();
    LOG.info("ACL 资源安全释放，程序退出。");
    return 0;
}