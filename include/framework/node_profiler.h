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

#include <atomic>
#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <limits>

namespace GryFlux
{
    /**
     * @brief 单个节点的性能统计数据
     *
     * 使用原子操作保证线程安全，避免锁竞争
     */
    struct NodeStats
    {
        std::atomic<uint64_t> count{0};      // 执行次数
        std::atomic<uint64_t> total_ns{0};   // 总耗时(纳秒)
        std::atomic<uint64_t> min_ns{UINT64_MAX}; // 最小耗时
        std::atomic<uint64_t> max_ns{0};     // 最大耗时

        NodeStats() = default;

        // 原子操作不可拷贝，需要自定义拷贝构造
        NodeStats(const NodeStats &other)
            : count(other.count.load()),
              total_ns(other.total_ns.load()),
              min_ns(other.min_ns.load()),
              max_ns(other.max_ns.load())
        {
        }

        NodeStats &operator=(const NodeStats &other)
        {
            if (this != &other)
            {
                count.store(other.count.load());
                total_ns.store(other.total_ns.load());
                min_ns.store(other.min_ns.load());
                max_ns.store(other.max_ns.load());
            }
            return *this;
        }

        /**
         * @brief 记录一次执行
         * @param duration_ns 执行耗时(纳秒)
         */
        void record(uint64_t duration_ns)
        {
            count.fetch_add(1, std::memory_order_relaxed);
            total_ns.fetch_add(duration_ns, std::memory_order_relaxed);

            // 更新最小值（使用 compare_exchange 保证原子性）
            uint64_t current_min = min_ns.load(std::memory_order_relaxed);
            while (duration_ns < current_min &&
                   !min_ns.compare_exchange_weak(current_min, duration_ns,
                                                   std::memory_order_relaxed))
            {
            }

            // 更新最大值
            uint64_t current_max = max_ns.load(std::memory_order_relaxed);
            while (duration_ns > current_max &&
                   !max_ns.compare_exchange_weak(current_max, duration_ns,
                                                   std::memory_order_relaxed))
            {
            }
        }

        /**
         * @brief 计算平均耗时
         * @return 平均耗时(纳秒)，如果count为0则返回0
         */
        uint64_t getAverageNs() const
        {
            uint64_t c = count.load(std::memory_order_relaxed);
            return c > 0 ? total_ns.load(std::memory_order_relaxed) / c : 0;
        }

        /**
         * @brief 重置统计数据
         */
        void reset()
        {
            count.store(0);
            total_ns.store(0);
            min_ns.store(UINT64_MAX);
            max_ns.store(0);
        }
    };

    /**
     * @brief 节点性能分析器（单例模式）
     *
     * 设计原则：
     * 1. 零侵入 - 不修改现有节点代码
     * 2. 低开销 - 使用无锁数据结构
     * 3. 可选 - 可通过编译宏关闭
     */
    class NodeProfiler
    {
    public:
        /**
         * @brief 获取全局单例
         */
        static NodeProfiler &getInstance()
        {
            static NodeProfiler instance;
            return instance;
        }

        /**
         * @brief 记录节点执行耗时
         * @param nodeName 节点名称
         * @param duration_ns 执行耗时(纳秒)
         */
        void recordNodeExecution(const std::string &nodeName, uint64_t duration_ns)
        {
            // 快速路径：尝试在已有的节点中记录
            {
                std::shared_lock<std::shared_mutex> lock(mutex_);
                auto it = stats_.find(nodeName);
                if (it != stats_.end())
                {
                    it->second.record(duration_ns);
                    return;
                }
            }

            // 慢路径：首次遇到该节点，需要插入
            {
                std::unique_lock<std::shared_mutex> lock(mutex_);
                stats_[nodeName].record(duration_ns);
            }
        }

        /**
         * @brief 获取所有节点的统计信息
         * @return 节点名称 -> 统计数据的映射
         */
        std::unordered_map<std::string, NodeStats> getStats() const
        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            return stats_;
        }

        /**
         * @brief 重置所有统计数据
         */
        void reset()
        {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            for (auto &[name, stats] : stats_)
            {
                stats.reset();
            }
        }

        /**
         * @brief 打印统计信息到标准输出
         */
        void printStats() const;

        /**
         * @brief 启用/禁用性能分析
         */
        void setEnabled(bool enabled) { enabled_.store(enabled); }

        /**
         * @brief 检查性能分析是否启用
         */
        bool isEnabled() const { return enabled_.load(); }

    private:
        NodeProfiler() = default;
        ~NodeProfiler() = default;

        // 禁止拷贝和移动
        NodeProfiler(const NodeProfiler &) = delete;
        NodeProfiler &operator=(const NodeProfiler &) = delete;

        mutable std::shared_mutex mutex_;
        std::unordered_map<std::string, NodeStats> stats_;
        std::atomic<bool> enabled_{true};
    };

    /**
     * @brief RAII 自动计时器
     *
     * 用法：
     *   {
     *       ScopedNodeTimer timer("nodeName");
     *       // ... 节点执行代码 ...
     *   } // 离开作用域时自动记录耗时
     */
    class ScopedNodeTimer
    {
    public:
        explicit ScopedNodeTimer(const std::string &nodeName)
            : nodeName_(nodeName),
              start_(std::chrono::high_resolution_clock::now())
        {
        }

        ~ScopedNodeTimer()
        {
            if (NodeProfiler::getInstance().isEnabled())
            {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
                NodeProfiler::getInstance().recordNodeExecution(nodeName_, duration);
            }
        }

    private:
        std::string nodeName_;
        std::chrono::high_resolution_clock::time_point start_;
    };

} // namespace GryFlux
