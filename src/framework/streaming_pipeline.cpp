/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Streaming Pipeline Implementation
 *************************************************************************************************************************/
#include "framework/streaming_pipeline.h"
#include "framework/node_profiler.h"
#include "utils/logger.h"
#include <chrono>

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
        NodeProfiler::getInstance().printStats();
    }

    void StreamingPipeline::resetProfilingStats()
    {
        NodeProfiler::getInstance().reset();
    }

    void StreamingPipeline::setProfilingEnabled(bool enabled)
    {
        NodeProfiler::getInstance().setEnabled(enabled);
    }

} // namespace GryFlux
