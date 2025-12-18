/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Async Pipeline Implementation
 *************************************************************************************************************************/
#include "framework/async_pipeline.h"
#include "framework/profiler/profiling.h"
#include "utils/logger.h"
#include <chrono>
#include <unordered_map>

namespace GryFlux
{

    AsyncPipeline::AsyncPipeline(std::shared_ptr<DataSource> source,
                                 std::shared_ptr<GraphTemplate> graphTemplate,
                                 std::shared_ptr<ResourcePool> resourcePool,
                                 std::shared_ptr<DataConsumer> consumer,
                                 size_t threadPoolSize,
                                 size_t maxActivePackets)
        : source_(source), consumer_(consumer), running_(false), producerDone_(false)
    {
        processor_ = std::make_shared<AsyncGraphProcessor>(
            graphTemplate,
            resourcePool,
            threadPoolSize,
            maxActivePackets);

        LOG.info("AsyncPipeline created");
    }

    AsyncPipeline::~AsyncPipeline()
    {
        stop();
    }

    void AsyncPipeline::run()
    {
        if (running_)
        {
            LOG.warning("AsyncPipeline already running");
            return;
        }

        running_ = true;
        producerDone_ = false;

        processor_->start();

        producerThread_ = std::thread(&AsyncPipeline::producerThread, this);
        consumerThread_ = std::thread(&AsyncPipeline::consumerThread, this);

        LOG.info("AsyncPipeline started");

        if (producerThread_.joinable())
        {
            producerThread_.join();
        }

        if (consumerThread_.joinable())
        {
            consumerThread_.join();
        }

        processor_->stop();

        running_ = false;
        LOG.info("AsyncPipeline completed");
    }

    void AsyncPipeline::stop()
    {
        if (!running_)
        {
            return;
        }

        LOG.info("Stopping AsyncPipeline...");
        running_ = false;

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

    void AsyncPipeline::producerThread()
    {
        LOG.info("Producer thread started");

        size_t producedCount = 0;

        while (running_ && source_->hasMore())
        {
            while (running_ && processor_->getActivePacketCount() >= processor_->getMaxActivePackets())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            if (!running_)
            {
                break;
            }

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

    void AsyncPipeline::consumerThread()
    {
        LOG.info("Consumer thread started");

        size_t consumedCount = 0;

        while (running_)
        {
            auto packet = processor_->tryGetOutput();

            if (packet)
            {
                consumer_->consume(std::move(packet));
                consumedCount++;
            }
            else
            {
                if (producerDone_ && processor_->getActivePacketCount() == 0)
                {
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        LOG.info("Consumer thread completed, consumed %zu packets", consumedCount);
    }

    void AsyncPipeline::printProfilingStats() const
    {
        if constexpr (!Profiling::kBuildProfiling)
        {
            LOG.error("当前构建未启用 graph profiling（Profiling::kBuildProfiling=false），跳过统计输出。");
            return;
        }

        auto events = Profiling::snapshotEvents();

        if (events.empty())
        {
            LOG.error("没有可用的 profiler 数据（是否已开启？）");
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
            case Profiling::EventType::Scheduled:
                entry.scheduled++;
                break;
            case Profiling::EventType::Started:
                entry.started++;
                break;
            case Profiling::EventType::Finished:
                entry.finished++;
                entry.totalDuration += evt.durationNs;
                break;
            case Profiling::EventType::Failed:
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

    void AsyncPipeline::resetProfilingStats()
    {
        if constexpr (Profiling::kBuildProfiling)
        {
            Profiling::reset();
            LOG.info("Profiler 数据已重置");
        }
        else
        {
            LOG.error("当前构建未启用 graph profiling（Profiling::kBuildProfiling=false），忽略 reset 请求。");
        }
    }

    void AsyncPipeline::setProfilingEnabled(bool enabled)
    {
        if constexpr (Profiling::kBuildProfiling)
        {
            Profiling::setEnabled(enabled);
            LOG.info("Profiler 已%s", enabled ? "启用" : "关闭");
        }
        else
        {
            (void)enabled;
            LOG.error("当前构建未启用 graph profiling（Profiling::kBuildProfiling=false），忽略 setEnabled 请求。");
        }
    }

    void AsyncPipeline::dumpProfilingTimeline(const std::string &filePath) const
    {
        if constexpr (Profiling::kBuildProfiling)
        {
            Profiling::dumpTimelineJson(filePath);
            LOG.info("Profiler 时间线已导出至 %s", filePath.c_str());
        }
        else
        {
            (void)filePath;
            LOG.error("当前构建未启用 graph profiling（Profiling::kBuildProfiling=false），跳过时间线导出。");
        }
    }

} // namespace GryFlux
