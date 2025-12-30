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

#include <vector>
#include <atomic>
#include <cstdint>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <stdexcept>
#include <type_traits>
#include "utils/noncopyable.h"

namespace GryFlux
{

    // 线程池实现
    class ThreadPool
        : private NonCopyableNonMovable
    {
    public:
        // 任务调度优先级（数值越小越优先）
        // 在图调度场景中，通常直接使用 DataPacket::getIdx() 作为 priority，
        // 以确保较小 idx 的数据包优先完成。
        using Priority = uint64_t;

        explicit ThreadPool(size_t numThreads);
        ~ThreadPool();

        // 提交任务到线程池
        template <class F>
        auto enqueue(F &&f) -> std::future<typename std::result_of<F()>::type>
        {
            return enqueue(0, std::forward<F>(f));
        }

        // 提交带优先级的任务到线程池（priority 越小越优先）
        template <class F>
        auto enqueue(Priority priority, F &&f) -> std::future<typename std::result_of<F()>::type>
        {
            using return_type = typename std::result_of<F()>::type;

            auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
            std::future<return_type> res = task->get_future();

            {
                std::unique_lock<std::mutex> lock(queueMutex_);
                if (stop_)
                {
                    throw std::runtime_error("enqueue on stopped ThreadPool");
                }
                TaskItem item;
                item.priority = priority;
                item.seq = nextSeq_.fetch_add(1, std::memory_order_relaxed);
                item.fn = [task]()
                { (*task)(); };

                tasks_.push(std::move(item));
            }

            condition_.notify_one();
            return res;
        }

        // 批量提交同一优先级的任务到线程池（内部一次加锁 push 多个 TaskItem）
        void enqueueBatch(Priority priority, std::vector<std::function<void()>> tasks);

        // 获取线程池中的线程数量
        size_t getThreadCount() const
        {
            return workers_.size();
        }

        // 获取当前待处理任务数量
        size_t getTaskCount() const
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            return tasks_.size();
        }

    private:
        struct TaskItem
        {
            Priority priority = 0;
            uint64_t seq = 0;
            std::function<void()> fn;
        };

        struct TaskItemCompare
        {
            bool operator()(const TaskItem &a, const TaskItem &b) const
            {
                if (a.priority != b.priority)
                {
                    return a.priority > b.priority; // smaller priority first
                }
                return a.seq > b.seq; // older first
            }
        };

        std::vector<std::thread> workers_;
        std::priority_queue<TaskItem, std::vector<TaskItem>, TaskItemCompare> tasks_;
        std::atomic<uint64_t> nextSeq_{0};
        mutable std::mutex queueMutex_;
        std::condition_variable condition_;
        bool stop_;
    };
}
