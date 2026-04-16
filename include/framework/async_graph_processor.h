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
#pragma once

#include "graph_template.h"
#include "resource_pool.h"
#include "task_scheduler.h"
#include "utils/threadsafe_queue.h"
#include "utils/noncopyable.h"
#include <memory>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <atomic>

namespace GryFlux
{

    /**
     * @brief 异步图处理器 - 简化版
     *
     * 提供图执行的基础设施。数据包之间的并行通过 ThreadPool 和事件驱动调度实现，
     * 无需额外的 Worker 线程。
     *
     * 使用方式：
     * 1. 创建 AsyncGraphProcessor
     * 2. 用户创建 DataPacket
     * 3. 调用 submitPacket() 提交执行
     * 4. 调用 tryGetOutput() 获取结果
     */
    class AsyncGraphProcessor
        : private NonCopyableNonMovable
    {
    public:
        /**
         * @brief 构造函数
         *
         * @param graphTemplate 图模板
         * @param resourcePool 资源池
         * @param threadPoolSize 线程池大小（0表示使用硬件并发数）
         * @param maxActivePackets 最大活跃数据包数（0表示自动：threadPoolSize - 1，且最小为 1）
         */
        AsyncGraphProcessor(std::shared_ptr<GraphTemplate> graphTemplate,
                            std::shared_ptr<ResourcePool> resourcePool,
                            size_t threadPoolSize = 0,
                            size_t maxActivePackets = 0);

        ~AsyncGraphProcessor();

        /**
         * @brief 启动处理器
         */
        void start();

        /**
         * @brief 停止处理器
         */
        void stop();

        /**
         * @brief 直接提交数据包执行（推荐方式）
         *
         * 用户创建数据包后直接提交执行，数据包之间通过线程池自动并行。
         * 所有权转移给框架。
         *
         * @param packet 数据包（unique_ptr，转移所有权）
         */
        void submitPacket(std::unique_ptr<DataPacket> packet);

        /**
         * @brief 获取完成的数据包（非阻塞）
         *
         * 获取完成的数据包，所有权转移给调用者。
         *
         * @return 完成的数据包（unique_ptr），如果无可用输出则返回 nullptr
         */
        std::unique_ptr<DataPacket> tryGetOutput();

        /**
         * @brief 获取完成的数据包（超时阻塞）
         *
         * 在给定超时时间内等待输出队列有数据可取，超时返回 false。
         *
         * @param packet 输出参数，成功时持有数据包所有权
         * @param timeout 超时时间
         * @return true 成功获取数据包
         * @return false 超时未获取到数据包
         */
        bool waitForOutput(std::unique_ptr<DataPacket> &packet, std::chrono::milliseconds timeout);

        /**
         * @brief 获取输出队列大小
         */
        size_t getOutputQueueSize() const;

        /**
         * @brief 获取当前活跃数据包数量
         */
        size_t getActivePacketCount() const;

        /**
         * @brief 获取最大活跃数据包数量
         */
        size_t getMaxActivePackets() const { return maxActivePackets_; }

    private:
        std::shared_ptr<GraphTemplate> graphTemplate_;
        std::shared_ptr<TaskScheduler> scheduler_;
        std::shared_ptr<ThreadPool> threadPool_;

        threadsafe_queue<std::unique_ptr<DataPacket>> outputQueue_;

        // 活跃数据包的所有权管理
        std::unordered_map<DataPacket*, std::unique_ptr<DataPacket>> activePackets_;
        mutable std::mutex activePacketsMutex_;

        size_t maxActivePackets_;  // 最大活跃数据包数
        std::atomic<bool> isRunning_;
    };

} // namespace GryFlux
