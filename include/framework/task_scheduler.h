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

#include <unordered_map>
#include <string>
#include <memory>
#include <future>
#include "framework/task_node.h"
#include "framework/thread_pool.h"

namespace GryFlux
{

    // 任务调度器
    class TaskScheduler
    {
    public:
        explicit TaskScheduler(size_t numThreads = 0);

        void addTask(std::shared_ptr<TaskNode> task);
        std::shared_ptr<TaskNode> getTask(const std::string &id);
        std::shared_ptr<DataObject> execute(const std::string &outputTaskId);

        // 清除所有任务
        void clear();
        
        // 获取所有任务的执行时间统计
        std::unordered_map<std::string, double> getTaskExecutionTimes() const;

    private:
        void executeTask(std::shared_ptr<TaskNode> task);

        ThreadPool threadPool_;
        std::unordered_map<std::string, std::shared_ptr<TaskNode>> tasks_;
    };

} // namespace GryFlux