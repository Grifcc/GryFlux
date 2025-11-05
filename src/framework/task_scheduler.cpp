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
#include "framework/profiler/graph_profiler.h"
#include "utils/logger.h"
#include <deque>

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
        auto &tmpl = packet->executionState_.graphTemplate;
        const auto &node = tmpl->getNode(nodeIndex);

        auto &profiler = GraphProfiler::instance();
        if (profiler.isEnabled())
        {
            profiler.recordNodeScheduled(packet, node.nodeId);
        }

        threadPool_->enqueue([this, packet, nodeIndex]()
                              { executeNodeChain(packet, nodeIndex); });
    }

    void TaskScheduler::executeNodeChain(DataPacket *packet, size_t nodeIndex)
    {
        auto &tmpl = packet->executionState_.graphTemplate;
        auto &profiler = GraphProfiler::instance();
        bool profilerEnabled = profiler.isEnabled();

        std::deque<size_t> readyQueue;
        readyQueue.push_back(nodeIndex);

        while (!readyQueue.empty())
        {
            size_t currentIndex = readyQueue.front();
            readyQueue.pop_front();

            auto &node = tmpl->getNode(currentIndex);
            std::shared_ptr<Context> ctx;

            if (!node.resourceTypeName.empty())
            {
                ctx = resourcePool_->acquire(node.resourceTypeName, resourceAcquireTimeout_);

                if (!ctx)
                {
                    LOG.error("Failed to acquire resource '%s' for node '%s' (index %zu)",
                              node.resourceTypeName.c_str(), node.nodeId.c_str(), currentIndex);

                    if (profilerEnabled)
                    {
                        profiler.recordNodeFailed(packet, node.nodeId, 0);
                    }

                    onNodeFailed(packet, currentIndex);
                    return;
                }
            }

            GraphProfiler::NodeExecutionScope execScope(packet, node.nodeId);

            try
            {
                if (ctx)
                {
                    node.taskFunc(*packet, *ctx);
                }
                else
                {
                    node.taskFunc(*packet, None::instance());
                }

                packet->markNodeCompleted(currentIndex);

                if (ctx)
                {
                    resourcePool_->release(node.resourceTypeName, ctx);
                }

                bool inlineAssigned = false;
                for (size_t succIdx : node.successorIndices)
                {
                    packet->notifyPredecessorCompleted(succIdx);

                    if (packet->tryMarkNodeReady(succIdx))
                    {
                        if (!inlineAssigned)
                        {
                            if (profilerEnabled)
                            {
                                profiler.recordNodeScheduled(packet, tmpl->getNode(succIdx).nodeId);
                            }
                            readyQueue.push_back(succIdx);
                            inlineAssigned = true;
                        }
                        else
                        {
                            scheduleNode(packet, succIdx);
                        }
                    }
                }

                if (currentIndex == tmpl->getOutputNodeIndex())
                {
                    onGraphCompleted(packet);
                }
            }
            catch (const std::exception &e)
            {
                execScope.markFailed();

                LOG.error("Node '%s' (index %zu) execution failed: %s",
                          node.nodeId.c_str(), currentIndex, e.what());

                if (ctx)
                {
                    resourcePool_->release(node.resourceTypeName, ctx);
                }

                onNodeFailed(packet, currentIndex);
                return;
            }
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
