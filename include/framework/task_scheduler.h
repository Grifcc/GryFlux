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
#pragma once

#include "resource_pool.h"
#include "data_packet.h"
#include "thread_pool.h"
#include <memory>
#include <functional>
#include <chrono>

namespace GryFlux
{

    /**
     * @brief 任务调度器 - 无状态的执行器
     *
     * 执行节点任务，管理资源获取，触发后继节点调度。
     */
    class TaskScheduler
    {
    public:
        TaskScheduler(std::shared_ptr<ResourcePool> resourcePool,
                      std::shared_ptr<ThreadPool> threadPool);

        ~TaskScheduler() = default;

        // 禁止拷贝和赋值
        TaskScheduler(const TaskScheduler &) = delete;
        TaskScheduler &operator=(const TaskScheduler &) = delete;

        /**
         * @brief 设置数据包完成回调
         *
         * @param callback 回调函数
         */
        void setCompletionCallback(std::function<void(DataPacket *)> callback);

        /**
         * @brief 设置资源获取超时时间
         *
         * @param timeout 超时时间
         */
        void setResourceAcquireTimeout(std::chrono::milliseconds timeout);

        /**
         * @brief 调度节点（提交到线程池）
         *
         * @param packet 数据包
         * @param nodeIndex 节点索引
         */
        void scheduleNode(DataPacket *packet, size_t nodeIndex);

    private:
        /**
         * @brief 在当前线程执行节点及其内联后继
         *
         * @param packet 数据包
         * @param nodeIndex 起始节点索引
         */
        void executeNodeChain(DataPacket *packet, size_t nodeIndex);

        /**
         * @brief 节点执行失败回调
         *
         * @param packet 数据包
         * @param nodeIndex 节点索引
         */
        void onNodeFailed(DataPacket *packet, size_t nodeIndex);

        /**
         * @brief 图执行完成回调
         *
         * @param packet 数据包
         */
        void onGraphCompleted(DataPacket *packet);

        std::shared_ptr<ResourcePool> resourcePool_;
        std::shared_ptr<ThreadPool> threadPool_;
        std::chrono::milliseconds resourceAcquireTimeout_;
        std::function<void(DataPacket *)> completionCallback_;
    };

} // namespace GryFlux
