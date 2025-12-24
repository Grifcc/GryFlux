/*************************************************************************************************************************
 * Copyright 2025 Grifcc & Sunhaihua1
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#include "framework/async_graph_processor.h"
#include "utils/logger.h"

namespace GryFlux
{

    AsyncGraphProcessor::AsyncGraphProcessor(std::shared_ptr<GraphTemplate> graphTemplate,
                                               std::shared_ptr<ResourcePool> resourcePool,
                                               size_t threadPoolSize,
                                               size_t maxActivePackets)
        : graphTemplate_(graphTemplate), isRunning_(false)
    {
        // 确定线程池大小
        size_t poolSize = threadPoolSize;
        if (poolSize == 0)
        {
            poolSize = std::thread::hardware_concurrency();
            if (poolSize == 0)
            {
                poolSize = 8; // 默认值
            }
        }

        // 确定最大活跃数据包数（默认为线程池 - 1）
        if (maxActivePackets == 0)
        {
            maxActivePackets_ = (poolSize > 1) ? (poolSize - 1) : 1;
        }
        else
        {
            maxActivePackets_ = maxActivePackets;
        }
        if (maxActivePackets_ == 0)
        {
            maxActivePackets_ = 1;
        }

        // 创建线程池
        threadPool_ = std::make_shared<ThreadPool>(poolSize);

        // 创建调度器
        scheduler_ = std::make_shared<TaskScheduler>(resourcePool, threadPool_);

        // 设置完成回调 - 从 activePackets_ 取回所有权并放入 outputQueue
        scheduler_->setCompletionCallback([this](DataPacket *packet)
                                           {
            std::lock_guard<std::mutex> lock(activePacketsMutex_);
            auto it = activePackets_.find(packet);
            if (it != activePackets_.end()) {
                outputQueue_.push(std::move(it->second));
                activePackets_.erase(it);
            } });

        LOG.info("AsyncGraphProcessor created with thread pool size %zu, max active packets %zu",
                 poolSize, maxActivePackets_);
    }

    AsyncGraphProcessor::~AsyncGraphProcessor()
    {
        stop();
    }

    void AsyncGraphProcessor::start()
    {
        if (isRunning_)
        {
            LOG.warning("AsyncGraphProcessor already running");
            return;
        }

        isRunning_ = true;
        LOG.info("AsyncGraphProcessor started");
    }

    void AsyncGraphProcessor::stop()
    {
        if (!isRunning_)
        {
            return;
        }

        isRunning_ = false;
        LOG.info("AsyncGraphProcessor stopped");
    }

    void AsyncGraphProcessor::submitPacket(std::unique_ptr<DataPacket> packet)
    {
        if (!isRunning_)
        {
            LOG.error("Cannot submit packet: AsyncGraphProcessor not running");
            return;
        }

        if (!packet)
        {
            LOG.error("Cannot submit null packet");
            return;
        }

        // 初始化执行状态
        packet->initializeExecution(graphTemplate_);

        // 取出裸指针用于调度，同时保留所有权在 activePackets_
        DataPacket *rawPtr = packet.get();
        {
            std::lock_guard<std::mutex> lock(activePacketsMutex_);
            activePackets_[rawPtr] = std::move(packet);
        }

        // 调度输入节点（索引0）
        // 后续所有节点会通过事件驱动自动调度
        scheduler_->scheduleNode(rawPtr, 0);
    }

    std::unique_ptr<DataPacket> AsyncGraphProcessor::tryGetOutput()
    {
        std::unique_ptr<DataPacket> result;
        outputQueue_.try_pop(result);
        return result;
    }

    size_t AsyncGraphProcessor::getOutputQueueSize() const
    {
        return static_cast<size_t>(outputQueue_.size());
    }

    size_t AsyncGraphProcessor::getActivePacketCount() const
    {
        std::lock_guard<std::mutex> lock(activePacketsMutex_);
        return activePackets_.size();
    }

} // namespace GryFlux
