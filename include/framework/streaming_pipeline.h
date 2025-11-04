/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Streaming Pipeline
 *************************************************************************************************************************/
#ifndef GRYFLUX_STREAMING_PIPELINE_H
#define GRYFLUX_STREAMING_PIPELINE_H

#include "data_source.h"
#include "data_consumer.h"
#include "async_graph_processor.h"
#include "graph_template.h"
#include "resource_pool.h"
#include <memory>
#include <thread>
#include <atomic>

namespace GryFlux
{

/**
 * @brief 流式处理管道 - Source → Graph → Consumer
 *
 * 封装完整的流式处理流程：
 * - Source: 持续产生数据
 * - Graph: 处理数据（通过 AsyncGraphProcessor）
 * - Consumer: 消费结果
 *
 * 自动管理：
 * - 生产者线程
 * - 消费者线程
 * - 背压控制（防止内存爆炸）
 * - 线程生命周期
 *
 * @example
 * @code
 * auto source = std::make_shared<VideoSource>("input.mp4");
 * auto consumer = std::make_shared<VideoWriter>("output.mp4");
 *
 * StreamingPipeline pipeline(source, graphTemplate, resourcePool, consumer);
 * pipeline.run();  // 阻塞直到处理完成
 * @endcode
 */
class StreamingPipeline
{
public:
    /**
     * @brief 构造函数
     *
     * @param source 数据源
     * @param graphTemplate 图模板
     * @param resourcePool 资源池
     * @param consumer 数据消费者
     * @param threadPoolSize 线程池大小（0表示自动）
     * @param maxActivePackets 最大活跃数据包数（0表示自动：threadPoolSize × 2）
     */
    StreamingPipeline(std::shared_ptr<DataSource> source,
                      std::shared_ptr<GraphTemplate> graphTemplate,
                      std::shared_ptr<ResourcePool> resourcePool,
                      std::shared_ptr<DataConsumer> consumer,
                      size_t threadPoolSize = 0,
                      size_t maxActivePackets = 0);

    ~StreamingPipeline();

    // 禁止拷贝和移动
    StreamingPipeline(const StreamingPipeline &) = delete;
    StreamingPipeline &operator=(const StreamingPipeline &) = delete;
    StreamingPipeline(StreamingPipeline &&) = delete;
    StreamingPipeline &operator=(StreamingPipeline &&) = delete;

    /**
     * @brief 运行管道（阻塞直到完成）
     *
     * 启动生产者和消费者线程，处理所有数据直到 Source 耗尽。
     */
    void run();

    /**
     * @brief 停止管道
     *
     * 停止生产和消费，等待当前数据包处理完成。
     */
    void stop();

    /**
     * @brief 打印所有节点的性能统计信息
     *
     * 输出每个节点的执行次数、平均耗时、最小/最大耗时等。
     */
    void printProfilingStats() const;

    /**
     * @brief 重置性能统计数据
     */
    void resetProfilingStats();

    /**
     * @brief 启用/禁用性能分析
     * @param enabled true=启用, false=禁用
     */
    void setProfilingEnabled(bool enabled);

private:
    /**
     * @brief 生产者线程函数
     *
     * 从 Source 获取数据，提交到 processor，带背压控制。
     */
    void producerThread();

    /**
     * @brief 消费者线程函数
     *
     * 从 processor 获取结果，提交到 Consumer。
     */
    void consumerThread();

    std::shared_ptr<DataSource> source_;
    std::shared_ptr<DataConsumer> consumer_;
    std::shared_ptr<AsyncGraphProcessor> processor_;

    std::thread producerThread_;
    std::thread consumerThread_;

    std::atomic<bool> running_;
    std::atomic<bool> producerDone_;
};

} // namespace GryFlux

#endif // GRYFLUX_STREAMING_PIPELINE_H
