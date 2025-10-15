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

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "data_object.h"

namespace GryFlux
{

    /**
     * @brief 处理任务的基类
     * 所有计算节点任务都应该继承这个基类，并实现process方法
     */
    class ProcessingTask
    {
    public:
        ProcessingTask() {};
        virtual ~ProcessingTask() = default;

        /**
         * @brief 处理数据的核心方法
         * @param inputs 输入数据对象列表
         * @return 处理后的数据对象
         */
        virtual std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) = 0;

        /**
         * @brief 获取绑定到当前任务实例的函数对象
         * @return 处理函数
         */
        std::function<std::shared_ptr<DataObject>(const std::vector<std::shared_ptr<DataObject>> &)>
        getProcessFunction()
        {
            return [this](const std::vector<std::shared_ptr<DataObject>> &inputs)
            {
                return this->process(inputs);
            };
        }
    };
    // 任务实例对象池，负责预先创建并复用 ProcessingTask 实例
    class ProcessingTaskPool : public std::enable_shared_from_this<ProcessingTaskPool>
    {
    public:
        using TaskPtr = std::shared_ptr<ProcessingTask>;
        using Factory = std::function<TaskPtr()>;

        ProcessingTaskPool(size_t capacity, Factory factory);
        ~ProcessingTaskPool();

        ProcessingTaskPool(const ProcessingTaskPool &) = delete;
        ProcessingTaskPool &operator=(const ProcessingTaskPool &) = delete;

        TaskPtr acquire();
        void release(TaskPtr task);
        void shutdown();

        size_t capacity() const { return capacity_; }
        size_t available() const;

    private:
        void initialize();

        size_t capacity_;
        Factory factory_;
        std::vector<TaskPtr> allTasks_;
        std::deque<TaskPtr> idleTasks_;

        mutable std::mutex mutex_;
        std::condition_variable cv_;
        bool stopped_;
    };

    // 定义任务注册表类，用于管理所有处理任务
    class TaskRegistry
    {
    public:
        TaskRegistry() = default;
        ~TaskRegistry();

        TaskRegistry(const TaskRegistry &) = delete;
        TaskRegistry &operator=(const TaskRegistry &) = delete;

        // 注册任务池并返回任务ID，支持自定义实例数量
        template <typename T>
        std::string registerTask(const std::string &taskId)
        {
            return registerTaskWithCount<T>(taskId, 1);
        }

        template <typename T, typename First, typename... Rest>
        std::string registerTask(const std::string &taskId, First &&first, Rest &&...rest)
        {
            if constexpr (std::is_integral_v<std::decay_t<First>> && !std::is_same_v<std::decay_t<First>, bool>)
            {
                return registerTaskWithCount<T>(taskId,
                                                static_cast<size_t>(std::forward<First>(first)),
                                                std::forward<Rest>(rest)...);
            }
            else
            {
                return registerTaskWithCount<T>(taskId, 1, std::forward<First>(first), std::forward<Rest>(rest)...);
            }
        }

        // 获取任务处理函数
        std::function<std::shared_ptr<DataObject>(const std::vector<std::shared_ptr<DataObject>> &)> getProcessFunction(const std::string &taskId);

    private:
        template <typename T, typename... Args>
        std::string registerTaskWithCount(const std::string &taskId, size_t instanceCount, Args &&...args)
        {
            if (taskPools_.find(taskId) != taskPools_.end())
            {
                throw std::runtime_error("Task already registered: " + taskId);
            }

            if (instanceCount == 0)
            {
                instanceCount = 1;
            }

            auto params = std::make_shared<std::tuple<std::decay_t<Args>...>>(std::forward<Args>(args)...);

            auto factory = [params]() -> std::shared_ptr<ProcessingTask>
            {
                return std::apply([](auto &...ctorArgs)
                                  { return std::make_shared<T>(ctorArgs...); }, *params);
            };

            auto pool = std::make_shared<ProcessingTaskPool>(instanceCount, std::move(factory));
            taskPools_.emplace(taskId, std::move(pool));
            return taskId;
        }

        std::unordered_map<std::string, std::shared_ptr<ProcessingTaskPool>> taskPools_;
    };
} // namespace GryFlux
