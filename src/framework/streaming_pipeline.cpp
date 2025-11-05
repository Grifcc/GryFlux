/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Streaming Pipeline Implementation
 *************************************************************************************************************************/
#include "framework/streaming_pipeline.h"
#include "framework/profiler/graph_profiler.h"
#include "utils/logger.h"
#include <chrono>
#include <unordered_map>

namespace GryFlux
{

    StreamingPipeline::StreamingPipeline(std::shared_ptr<DataSource> source,
                                         std::shared_ptr<GraphTemplate> graphTemplate,
                                         std::shared_ptr<ResourcePool> resourcePool,
                                         std::shared_ptr<DataConsumer> consumer,
                                         size_t threadPoolSize,
                                         size_t maxActivePackets)
        : source_(source), consumer_(consumer), running_(false), producerDone_(false)
    {
        // 创建 AsyncGraphProcessor
        processor_ = std::make_shared<AsyncGraphProcessor>(
            graphTemplate,
            resourcePool,
            threadPoolSize,
            maxActivePackets);

        LOG.info("StreamingPipeline created");
    }

    StreamingPipeline::~StreamingPipeline()
    {
        stop();
    }

    void StreamingPipeline::run()
    {
        if (running_)
        {
            LOG.warning("StreamingPipeline already running");
            return;
        }

        running_ = true;
        producerDone_ = false;

        // 启动 processor
        processor_->start();

        // 启动生产者线程
        producerThread_ = std::thread(&StreamingPipeline::producerThread, this);

        // 启动消费者线程
        consumerThread_ = std::thread(&StreamingPipeline::consumerThread, this);

        LOG.info("StreamingPipeline started");

        // 等待生产者完成
        if (producerThread_.joinable())
        {
            producerThread_.join();
        }

        // 等待消费者完成
        if (consumerThread_.joinable())
        {
            consumerThread_.join();
        }

        // 停止 processor
        processor_->stop();

        running_ = false;
        LOG.info("StreamingPipeline completed");
    }

    void StreamingPipeline::stop()
    {
        if (!running_)
        {
            return;
        }

        LOG.info("Stopping StreamingPipeline...");
        running_ = false;

        // 等待线程结束
        if (producerThread_.joinable())
        {
            producerThread_.join();
        }

        if (consumerThread_.joinable())
        {
            consumerThread_.join();
        }

        processor_->stop();
    }

    void StreamingPipeline::producerThread()
    {
        LOG.info("Producer thread started");

        size_t producedCount = 0;

        while (running_ && source_->hasMore())
        {
            // 背压控制：等待有空位
            while (running_ && processor_->getActivePacketCount() >= processor_->getMaxActivePackets())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            if (!running_)
            {
                break;
            }

            // 生产数据包
            auto packet = source_->produce();
            if (packet)
            {
                processor_->submitPacket(std::move(packet));
                producedCount++;
            }
        }

        producerDone_ = true;
        LOG.info("Producer thread completed, produced %zu packets", producedCount);
    }

    void StreamingPipeline::consumerThread()
    {
        LOG.info("Consumer thread started");

        size_t consumedCount = 0;

        while (running_)
        {
            // 尝试获取输出
            auto packet = processor_->tryGetOutput();

            if (packet)
            {
                // 消费数据包
                consumer_->consume(std::move(packet));
                consumedCount++;
            }
            else
            {
                // 如果生产者完成且没有活跃数据包，退出
                if (producerDone_ && processor_->getActivePacketCount() == 0)
                {
                    break;
                }

                // 否则短暂休眠
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        LOG.info("Consumer thread completed, consumed %zu packets", consumedCount);
    }

    void StreamingPipeline::printProfilingStats() const
    {
        auto events = GraphProfiler::instance().snapshotEvents();

        if (events.empty())
        {
            LOG.info("没有可用的 profiler 数据（是否已开启？）");
            return;
        }

        struct Summary
        {
            size_t scheduled = 0;
            size_t started = 0;
            size_t finished = 0;
            size_t failed = 0;
            uint64_t totalDuration = 0;
        };

        std::unordered_map<std::string, Summary> summaryMap;

        for (const auto &evt : events)
        {
            auto &entry = summaryMap[evt.nodeId];

            switch (evt.type)
            {
            case GraphProfiler::EventType::Scheduled:
                entry.scheduled++;
                break;
            case GraphProfiler::EventType::Started:
                entry.started++;
                break;
            case GraphProfiler::EventType::Finished:
                entry.finished++;
                entry.totalDuration += evt.durationNs;
                break;
            case GraphProfiler::EventType::Failed:
                entry.failed++;
                entry.totalDuration += evt.durationNs;
                break;
            }
        }

        LOG.info("========= Profiling Summary =========");
        for (const auto &[node, entry] : summaryMap)
        {
            double avgMs = entry.finished > 0 ? (entry.totalDuration / static_cast<double>(entry.finished)) / 1'000'000.0 : 0.0;
            double totalMs = entry.totalDuration / 1'000'000.0;

            LOG.info("节点 %s => scheduled=%zu started=%zu finished=%zu failed=%zu avg=%.3f ms total=%.3f ms",
                     node.c_str(),
                     entry.scheduled,
                     entry.started,
                     entry.finished,
                     entry.failed,
                     avgMs,
                     totalMs);
        }
        LOG.info("=====================================");
    }

    void StreamingPipeline::resetProfilingStats()
    {
        GraphProfiler::instance().reset();
        LOG.info("Profiler 数据已重置");
    }

    void StreamingPipeline::setProfilingEnabled(bool enabled)
    {
        GraphProfiler::instance().setEnabled(enabled);
        LOG.info("Profiler 已%s", enabled ? "启用" : "关闭");
    }

    void StreamingPipeline::dumpProfilingTimeline(const std::string &filePath) const
    {
        GraphProfiler::instance().dumpTimelineJson(filePath);
        LOG.info("Profiler 时间线已导出至 %s", filePath.c_str());
    }

} // namespace GryFlux
