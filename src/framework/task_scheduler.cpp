/*************************************************************************************************************************
 * Copyright 2025 Grifcc
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
#include "framework/task_scheduler.h"
#include "framework/graph_template.h"
#include "framework/node_profiler.h"
#include "utils/logger.h"

namespace GryFlux
{

    TaskScheduler::TaskScheduler(std::shared_ptr<ResourcePool> resourcePool,
                                   std::shared_ptr<ThreadPool> threadPool)
        : resourcePool_(resourcePool), threadPool_(threadPool), resourceAcquireTimeout_(std::chrono::seconds(10))
    {
    }

    void TaskScheduler::setCompletionCallback(std::function<void(DataPacket *)> callback)
    {
        completionCallback_ = callback;
    }

    void TaskScheduler::setResourceAcquireTimeout(std::chrono::milliseconds timeout)
    {
        resourceAcquireTimeout_ = timeout;
    }

    void TaskScheduler::scheduleNode(DataPacket *packet, size_t nodeIndex)
    {
        // 统一调度策略：所有节点都提交到线程池
        threadPool_->enqueue([this, packet, nodeIndex]()
                              { executeNode(packet, nodeIndex); });
    }

    void TaskScheduler::executeNode(DataPacket *packet, size_t nodeIndex)
    {
        auto &tmpl = packet->executionState_.graphTemplate;
        auto &node = tmpl->getNode(nodeIndex);

        std::shared_ptr<Context> ctx;

        // 1. 如果需要资源，阻塞获取
        if (!node.resourceTypeName.empty())
        {
            ctx = resourcePool_->acquire(node.resourceTypeName, resourceAcquireTimeout_);

            if (!ctx)
            {
                LOG.error("Failed to acquire resource '%s' for node '%s' (index %zu)",
                          node.resourceTypeName.c_str(), node.nodeId.c_str(), nodeIndex);
                onNodeFailed(packet, nodeIndex);
                return;
            }
        }

        try
        {
            // 2. 执行任务（修改 packet 内的数据）
            // packet 传引用（借用语义）
            // ctx 传引用（防止误操作，CPU任务使用None）
            {
                // RAII 自动计时（性能分析）
                ScopedNodeTimer timer(node.nodeId);

                if (ctx)
                {
                    node.taskFunc(*packet, *ctx);
                }
                else
                {
                    node.taskFunc(*packet, None::instance());
                }
            }

            // 3. 标记完成
            packet->markNodeCompleted(nodeIndex);

            // 4. 释放资源（RAII）
            if (ctx)
            {
                resourcePool_->release(node.resourceTypeName, ctx);
            }

            // 5. 通知并调度所有后继（事件驱动核心）
            for (size_t succIdx : node.successorIndices)
            {
                packet->notifyPredecessorCompleted(succIdx);

                if (packet->tryMarkNodeReady(succIdx))
                {
                    scheduleNode(packet, succIdx); // 后继就绪，立即调度
                }
            }

            // 6. 检查是否是输出节点
            if (nodeIndex == tmpl->getOutputNodeIndex())
            {
                onGraphCompleted(packet);
            }
        }
        catch (const std::exception &e)
        {
            LOG.error("Node '%s' (index %zu) execution failed: %s",
                      node.nodeId.c_str(), nodeIndex, e.what());

            if (ctx)
            {
                resourcePool_->release(node.resourceTypeName, ctx);
            }

            onNodeFailed(packet, nodeIndex);
        }
    }

    void TaskScheduler::onNodeFailed(DataPacket *packet, size_t nodeIndex)
    {
        packet->executionState_.isGraphCompleted.store(true, std::memory_order_release);

        LOG.warning("Data packet failed at node index %zu, deleting packet", nodeIndex);

        // 失败的包直接删除
        delete packet;
    }

    void TaskScheduler::onGraphCompleted(DataPacket *packet)
    {
        if (completionCallback_)
        {
            completionCallback_(packet);
        }
    }

} // namespace GryFlux
