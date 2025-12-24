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
#include <deque>
#include <stdexcept>

namespace GryFlux
{
    TaskScheduler::TaskScheduler(std::shared_ptr<ResourcePool> resourcePool,
                                   std::shared_ptr<ThreadPool> threadPool)
        : resourcePool_(resourcePool), threadPool_(threadPool), resourceAcquireTimeout_(std::chrono::milliseconds(0))
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
        if (!packet || packet->executionState_.isGraphCompleted.load(std::memory_order_acquire))
        {
            return;
        }

        if constexpr (Profiling::kBuildProfiling)
        {
            Profiling::recordNodeScheduled(packet, packet->executionState_.graphTemplate->getTask(nodeIndex).nodeId);
        }

        threadPool_->enqueue([this, packet, nodeIndex]()
                              { executeNodeChain(packet, nodeIndex); });
    }

    void TaskScheduler::executeNodeChain(DataPacket *packet, size_t nodeIndex)
    {
        auto &tmpl = packet->executionState_.graphTemplate;

        std::deque<size_t> readyQueue;
        readyQueue.push_back(nodeIndex);

        while (!readyQueue.empty())
        {
            if (packet->executionState_.isGraphCompleted.load(std::memory_order_acquire))
            {
                return;
            }

            size_t currentIndex = readyQueue.front();
            readyQueue.pop_front();

            auto &node = tmpl->getTask(currentIndex);
            std::shared_ptr<Context> ctx;

            const bool packetFailed = packet->executionState_.hasFailed.load(std::memory_order_acquire);

            // If the packet has failed, we still fast-forward the DAG to keep graph integrity,
            // but we skip resource acquisition and node execution.
            if (!packetFailed && !node.resourceTypeName.empty())
            {
                ctx = resourcePool_->acquire(node.resourceTypeName,
                                             resourceAcquireTimeout_,
                                             &packet->executionState_.hasFailed);
            }

            Profiling::NodeScope execScope(packet, node.nodeId);

            if (!packetFailed && !packet->executionState_.hasFailed.load(std::memory_order_acquire))
            {
                if (!node.nodeImpl)
                {
                    execScope.markFailed();
                    LOG.error("Node '%s' (index %zu) implementation is null", node.nodeId.c_str(), currentIndex);
                    packet->markFailed();
                }
                else if (!node.resourceTypeName.empty() && !ctx)
                {
                    execScope.markFailed();
                    LOG.error("Failed to acquire resource '%s' for node '%s' (index %zu)",
                              node.resourceTypeName.c_str(), node.nodeId.c_str(), currentIndex);
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
                                  node.nodeId.c_str(), currentIndex, e.what());
                        packet->markFailed();
                    }
                }
            }

            // Always release the resource (if acquired), even if the packet has failed.
            if (ctx)
            {
                resourcePool_->release(node.resourceTypeName, ctx);
            }

            // Always advance the DAG so the output node can be reached.
            packet->markNodeCompleted(currentIndex);

            bool inlineAssigned = false;
            for (size_t succIdx : node.childIndices)
            {
                packet->notifyPredecessorCompleted(succIdx);

                if (packet->tryMarkNodeReady(succIdx))
                {
                    if (!inlineAssigned)
                    {
                        if constexpr (Profiling::kBuildProfiling)
                        {
                            Profiling::recordNodeScheduled(packet, tmpl->getTask(succIdx).nodeId);
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
    }

    void TaskScheduler::onGraphCompleted(DataPacket *packet)
    {
        if (completionCallback_)
        {
            completionCallback_(packet);
        }
    }

} // namespace GryFlux
