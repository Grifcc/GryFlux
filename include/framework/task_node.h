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

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <atomic>
#include <chrono>
#include <mutex>
#include "framework/data_object.h"

namespace GryFlux
{

    // 任务节点基类
    class TaskNode
    {
    public:
        using TaskId = std::string;

        TaskNode(TaskId id);
        virtual ~TaskNode() = default;

        TaskId getId() const;
        void addDependency(std::shared_ptr<TaskNode> node);
        const std::vector<std::shared_ptr<TaskNode>> &getDependencies() const;
        void setResult(std::shared_ptr<DataObject> result);
        std::shared_ptr<DataObject> getResult();
        bool isExecuted() const;

        virtual std::shared_ptr<DataObject> execute() = 0;
        void executeOnce(); //保证同一个任务不会被多次执行
        virtual bool isReady() const; // 添加isReady方法

        // 执行时间相关方法
        void startExecution();
        void endExecution();
        double getExecutionTimeMs() const;


    protected:
        TaskId id_;
        std::vector<std::shared_ptr<TaskNode>> dependencies_;
        std::shared_ptr<DataObject> result_;
        bool executed_;

        // 任务执行时间记录
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;
        std::chrono::time_point<std::chrono::high_resolution_clock> endTime_;
        double executionTimeMs_;
        
        // 互斥锁保护任务执行过程和共享数据访问，确保线程安全
        mutable std::recursive_mutex mutex_;

    };

    // 输入数据源节点
    class InputNode : public TaskNode
    {
    public:
        InputNode(TaskId id, std::shared_ptr<DataObject> data);
        std::shared_ptr<DataObject> execute() override;

    private:
        std::shared_ptr<DataObject> data_;
    };

    // 具有多个输入的任务
    class MultiInputTaskNode : public TaskNode
    {
    public:
        using ProcessFunction = std::function<std::shared_ptr<DataObject>(const std::vector<std::shared_ptr<DataObject>> &)>;

        MultiInputTaskNode(TaskId id, ProcessFunction func,
                           const std::vector<std::shared_ptr<TaskNode>> &inputs);
        std::shared_ptr<DataObject> execute() override;
        bool isReady() const override;

    private:
        ProcessFunction func_;
        std::vector<std::shared_ptr<TaskNode>> inputs_;
        std::shared_ptr<DataObject> result_;
    };

} // namespace GryFlux