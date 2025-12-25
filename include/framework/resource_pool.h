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

#include "context.h"
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <chrono>
#include "utils/noncopyable.h"

namespace GryFlux
{

    /**
     * @brief 资源池 - 管理有限的硬件资源
     *
     * 用于管理NPU、DPU、Tracker等硬件资源，控制并发度。
     * TODO: Tracker节点作为一个全局唯一上下文的特殊情况，目前需要抽象为单独的资源类型。
     */
    class ResourcePool
        : private NonCopyableNonMovable
    {
    public:
        ResourcePool() = default;
        ~ResourcePool() = default;

        /**
         * @brief 注册资源类型
         *
         * @param typeName 资源类型名称（如 "npu", "tracker" 等）
         * @param instances 资源实例列表
         * @param acquireTimeout 资源获取超时时间（0ms 表示无限等待）
         */
        void registerResourceType(const std::string &typeName,
                                   std::vector<std::shared_ptr<Context>> instances,
                                   std::chrono::milliseconds acquireTimeout = std::chrono::milliseconds(0));

        /**
         * @brief 设置某个资源类型的获取超时时间（0ms 表示无限等待）
         */
        void setAcquireTimeout(const std::string &typeName, std::chrono::milliseconds acquireTimeout);

        /**
         * @brief 获取某个资源类型的获取超时时间（未设置则返回 0ms）
         */
        std::chrono::milliseconds getAcquireTimeout(const std::string &typeName) const;

        /**
         * @brief 获取资源（阻塞直到可用或超时）
         *
         * @param typeName 资源类型名称
         * @param timeout 超时时间
         *        - timeout == 0ms：无限等待
         *        - timeout  > 0ms：等待直到超时返回 nullptr
         * @return 资源上下文，超时返回 nullptr
         */
        std::shared_ptr<Context> acquire(const std::string &typeName,
                                          std::chrono::milliseconds timeout = std::chrono::seconds(10));

        /**
         * @brief 获取资源（可取消）
         *
         * 用于“协作式取消”：当数据包进入 failed 状态后，等待资源的线程可尽快退出，避免无意义地占用资源/线程。
         *
         * @param typeName 资源类型名称
         * @param timeout 超时时间（0ms 表示无限等待，但仍会响应 cancelFlag）
         * @param cancelFlag 取消标志指针（不可为空）；若为 true 则提前返回 nullptr（不记录超时 warning）
         * @return 资源上下文，超时或取消返回 nullptr
         */
        std::shared_ptr<Context> acquire(const std::string &typeName,
                                         std::chrono::milliseconds timeout,
                                         const std::atomic<bool> *cancelFlag);

        /**
         * @brief 释放资源
         *
         * @param typeName 资源类型名称
         * @param context 资源上下文
         */
        void release(const std::string &typeName, std::shared_ptr<Context> context);

        /**
         * @brief 查询可用资源数量
         *
         * @param typeName 资源类型名称
         * @return 可用资源数量
         */
        size_t getAvailableCount(const std::string &typeName) const;

    private:
        struct ResourceTypePool
        {
            std::queue<std::shared_ptr<Context>> availableContexts;
            std::vector<std::shared_ptr<Context>> allContexts;
            std::chrono::milliseconds acquireTimeout{std::chrono::milliseconds(0)};
            std::mutex poolMutex;
            std::condition_variable availabilityCondition;
        };

        mutable std::mutex resourceRegistryMutex_;
        std::unordered_map<std::string, ResourceTypePool> resourceTypePools_;
    };

} // namespace GryFlux
