/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#pragma once

#include <condition_variable> // NOLINT
#include <memory>
#include <mutex> // NOLINT
#include <queue>
#include <utility>

template <typename T>
class threadsafe_queue
{
private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;

public:
    threadsafe_queue() {}

    // 支持拷贝语义
    void push(const T &data)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(data);
        condition_.notify_one();
    }

    // 支持移动语义
    void push(T &&data)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(data));
        condition_.notify_one();
    }

    // 阻塞等待数据
    void wait_and_pop(T &value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]
                        { return !queue_.empty(); });
        value = std::move(queue_.front());
        queue_.pop();
    }

    // 非阻塞获取数据（使用移动语义）
    bool try_pop(T &value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    int size() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    ~threadsafe_queue() {}
};
