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
#include "framework/data_packet.h"
#include "framework/graph_template.h"
#include <stdexcept>

namespace GryFlux
{

    void DataPacket::initializeExecution(std::shared_ptr<GraphTemplate> tmpl)
    {
        if (!tmpl)
        {
            throw std::invalid_argument("GraphTemplate is null");
        }

        if (tmpl->getNodeCount() == 0)
        {
            throw std::runtime_error("GraphTemplate has no nodes");
        }

        executionState_.graphTemplate = tmpl;
        executionState_.isGraphCompleted.store(false, std::memory_order_relaxed);
        executionState_.hasFailed.store(false, std::memory_order_relaxed);
        executionState_.inFlight.store(0, std::memory_order_relaxed);
        executionState_.completionNotified.store(false, std::memory_order_relaxed);

        // 使用 unique_ptr 避免移动/拷贝 atomic 成员
        executionState_.nodeStates.clear();
        executionState_.nodeStates.reserve(tmpl->getNodeCount());

        // 初始化每个节点的依赖计数
        for (size_t i = 0; i < tmpl->getNodeCount(); ++i)
        {
            auto &node = tmpl->getTask(i);

            auto state = std::make_unique<ExecutionState::NodeState>();
            state->unfinishedPredecessorCount.store(node.parentIndices.size(), std::memory_order_relaxed);
            state->hasBeenEnqueued.store(false, std::memory_order_relaxed);
            state->isCompleted.store(false, std::memory_order_relaxed);

            executionState_.nodeStates.push_back(std::move(state));
        }

        // 输入节点立即就绪（无依赖）
        executionState_.nodeStates[tmpl->getInputNodeIndex()]->unfinishedPredecessorCount.store(0, std::memory_order_relaxed);
    }

    bool DataPacket::tryMarkNodeReady(size_t nodeIndex)
    {
        auto &state = *executionState_.nodeStates[nodeIndex];

        // 检查依赖是否全部完成
        if (state.unfinishedPredecessorCount.load(std::memory_order_acquire) == 0)
        {
            bool expected = false;
            // CAS：只有一个线程能成功
            if (state.hasBeenEnqueued.compare_exchange_strong(
                    expected, true,
                    std::memory_order_acq_rel))
            {
                return true; // 成功标记，应该调度此节点
            }
        }
        return false;
    }

    void DataPacket::notifyPredecessorCompleted(size_t nodeIndex)
    {
        auto &state = *executionState_.nodeStates[nodeIndex];
        state.unfinishedPredecessorCount.fetch_sub(1, std::memory_order_acq_rel);
    }

    void DataPacket::markNodeCompleted(size_t nodeIndex)
    {
        executionState_.nodeStates[nodeIndex]->isCompleted.store(true, std::memory_order_release);

        // 检查是否是输出节点
        if (nodeIndex == executionState_.graphTemplate->getOutputNodeIndex())
        {
            executionState_.isGraphCompleted.store(true, std::memory_order_release);
        }
    }

    bool DataPacket::isCompleted() const
    {
        return executionState_.isGraphCompleted.load(std::memory_order_acquire);
    }

    bool DataPacket::isFailed() const
    {
        return executionState_.hasFailed.load(std::memory_order_acquire);
    }

    void DataPacket::markFailed()
    {
        executionState_.hasFailed.store(true, std::memory_order_release);
    }

    void DataPacket::markTaskScheduled()
    {
        executionState_.inFlight.fetch_add(1, std::memory_order_acq_rel);
    }

    bool DataPacket::markTaskFinished()
    {
        uint32_t scheduledCount = executionState_.inFlight.load(std::memory_order_acquire);
        while (true)
        {
            if (scheduledCount == 0)
            {
                return false;
            }

            if (executionState_.inFlight.compare_exchange_weak(
                    scheduledCount,
                    scheduledCount - 1,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                break;
            }
        }

        const uint32_t remaining = scheduledCount - 1;
        if (remaining != 0)
        {
            return false;
        }

        if (!executionState_.isGraphCompleted.load(std::memory_order_acquire))
        {
            return false;
        }

        bool expected = false;
        return executionState_.completionNotified.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel);
    }

} // namespace GryFlux
