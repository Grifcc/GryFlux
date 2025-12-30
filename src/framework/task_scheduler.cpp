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
#include "framework/task_scheduler.h"
#include "framework/graph_template.h"
#include "framework/node_base.h"
#include "framework/profiler/profiling.h"
#include "utils/logger.h"
#include <vector>
#include <stdexcept>

namespace GryFlux
{
    TaskScheduler::TaskScheduler(std::shared_ptr<ResourcePool> resourcePool,
                                   std::shared_ptr<ThreadPool> threadPool)
        : resourcePool_(resourcePool), threadPool_(threadPool)
    {
    }

    void TaskScheduler::setCompletionCallback(std::function<void(DataPacket *)> callback)
    {
        completionCallback_ = callback;
    }

    void TaskScheduler::scheduleNode(DataPacket *packet, size_t nodeIndex, ThreadPool::Priority priority)
    {
        if (!packet || packet->executionState_.isGraphCompleted.load(std::memory_order_acquire))
        {
            return;
        }

        // 默认使用 packet idx 作为主优先级，确保小 idx 的包优先完成。
        const ThreadPool::Priority packetPriority =
            (priority != 0) ? priority : static_cast<ThreadPool::Priority>(packet->getIdx());

        if constexpr (Profiling::kBuildProfiling)
        {
            Profiling::recordNodeScheduled(packet, packet->executionState_.graphTemplate->getTask(nodeIndex).nodeId);
        }

        threadPool_->enqueue(packetPriority, [this, packet, nodeIndex, packetPriority]()
                             { executeNode(packet, nodeIndex, packetPriority); });
    }

    void TaskScheduler::executeNode(DataPacket *packet, size_t nodeIndex, ThreadPool::Priority priority)
    {
        auto &tmpl = packet->executionState_.graphTemplate;
        auto &node = tmpl->getTask(nodeIndex);

        if (packet->executionState_.isGraphCompleted.load(std::memory_order_acquire))
        {
            return;
        }

        std::shared_ptr<Context> ctx;

        const bool packetFailed = packet->executionState_.hasFailed.load(std::memory_order_acquire);

        // 如果 packet 已失败：跳过资源获取与执行，但仍推进 DAG，保证最终能到达 output 节点。
        if (packetFailed)
        {
            if constexpr (Profiling::kBuildProfiling)
            {
                Profiling::recordNodeSkipped(packet, node.nodeId);
            }
        }
        else
        {
            if (!node.resourceTypeName.empty())
            {
                const auto timeout = resourcePool_ ? resourcePool_->getAcquireTimeout(node.resourceTypeName)
                                                   : std::chrono::milliseconds(0);
                ctx = resourcePool_->acquire(node.resourceTypeName, timeout, &packet->executionState_.hasFailed);
            }

            Profiling::NodeScope execScope(packet, node.nodeId);

            if (!packet->executionState_.hasFailed.load(std::memory_order_acquire))
            {
                if (!node.nodeImpl)
                {
                    execScope.markFailed();
                    LOG.error("Node '%s' (index %zu) implementation is null", node.nodeId.c_str(), nodeIndex);
                    packet->markFailed();
                }
                else if (!node.resourceTypeName.empty() && !ctx)
                {
                    execScope.markFailed();
                    LOG.error("Failed to acquire resource '%s' for node '%s' (index %zu)",
                              node.resourceTypeName.c_str(), node.nodeId.c_str(), nodeIndex);
                    packet->markFailed();
                }
                else
                {
                    try
                    {
                        if (ctx)
                        {
                            node.nodeImpl->execute(*packet, *ctx);
                        }
                        else
                        {
                            node.nodeImpl->execute(*packet, None::instance());
                        }
                    }
                    catch (const std::exception &e)
                    {
                        execScope.markFailed();
                        LOG.error("Node '%s' (index %zu) execution failed: %s",
                                  node.nodeId.c_str(), nodeIndex, e.what());
                        packet->markFailed();
                    }
                }
            }
        }

        // Always release the resource (if acquired), even if the packet has failed.
        if (ctx)
        {
            resourcePool_->release(node.resourceTypeName, ctx);
        }

        // Always advance the DAG so the output node can be reached.
        packet->markNodeCompleted(nodeIndex);

        std::vector<std::function<void()>> successorTasks;
        successorTasks.reserve(node.childIndices.size());

        for (size_t succIdx : node.childIndices)
        {
            packet->notifyPredecessorCompleted(succIdx);

            if (packet->tryMarkNodeReady(succIdx))
            {
                if constexpr (Profiling::kBuildProfiling)
                {
                    Profiling::recordNodeScheduled(packet, tmpl->getTask(succIdx).nodeId);
                }

                successorTasks.emplace_back([this, packet, succIdx, priority]()
                                            { executeNode(packet, succIdx, priority); });
            }
        }

        if (!successorTasks.empty())
        {
            threadPool_->enqueueBatch(priority, std::move(successorTasks));
        }

        if (nodeIndex == tmpl->getOutputNodeIndex())
        {
            onGraphCompleted(packet);
        }
    }

    void TaskScheduler::onGraphCompleted(DataPacket *packet)
    {
        if (completionCallback_)
        {
            completionCallback_(packet);
        }
    }

} // namespace GryFlux
