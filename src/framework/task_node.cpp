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
#include "framework/task_node.h"
#include <algorithm>
#include <mutex>
#include "utils/logger.h"

namespace GryFlux
{

    // TaskNode基类实现
    TaskNode::TaskNode(TaskId id) : id_(id), executed_(false), executionTimeMs_(0.0) {}

    TaskNode::TaskId TaskNode::getId() const
    {
        return id_;
    }

    void TaskNode::addDependency(std::shared_ptr<TaskNode> node)
    {
        if (node)
        {
            dependencies_.push_back(node);
        }
    }

    const std::vector<std::shared_ptr<TaskNode>> &TaskNode::getDependencies() const
    {
        return dependencies_;
    }

    void TaskNode::setResult(std::shared_ptr<DataObject> result)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        result_ = result;
        executed_ = true;
    }

    std::shared_ptr<DataObject> TaskNode::getResult()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        return result_;
    }

    bool TaskNode::isExecuted() const
    {
        // 使用互斥锁保护任务执行过程
        std::lock_guard<std::recursive_mutex> lock(mutex_);

        return executed_;
    }

    bool TaskNode::isReady() const
    {
        for (const auto &dependency : dependencies_)
        {
            // 添加空指针检查
            if (!dependency || !dependency->isExecuted())
            {
                return false;
            }
        }
        return true;
    }

    void TaskNode::startExecution()
    {
        try {
            startTime_ = std::chrono::high_resolution_clock::now();
        } catch (const std::exception& e) {
            LOG.error("Exception in startExecution: %s", e.what());
        }
    }

    void TaskNode::endExecution()
    {
        try {
            endTime_ = std::chrono::high_resolution_clock::now();
            executionTimeMs_ = std::chrono::duration<double, std::milli>(endTime_ - startTime_).count();
        } catch (const std::exception& e) {
            LOG.error("Exception in endExecution: %s", e.what());
            executionTimeMs_ = 0.0;  // 默认值以防出错
        }
    }

    void TaskNode::executeOnce()
    {
        // 使用互斥锁保护任务执行过程
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        
        // 再次检查是否已执行，以防在等待依赖时被其他线程执行
        if (isExecuted()) {
            return;
        }

        // 记录任务开始执行时间
        startExecution();
        
        // 执行当前任务
        auto result = execute();
        
        // 记录任务结束执行时间
        endExecution();

        // 设置结果（必须在记录结束时间之后）
        setResult(result);
        
        // 记录执行时间
        LOG.debug("Task [%s] executed in %.3f ms", getId().c_str(), getExecutionTimeMs());

    }

    double TaskNode::getExecutionTimeMs() const
    {
        // 使用互斥锁保护任务执行过程
        std::lock_guard<std::recursive_mutex> lock(mutex_);

        return executionTimeMs_;
    }

    // InputNode实现
    InputNode::InputNode(TaskId id, std::shared_ptr<DataObject> data)
        : TaskNode(id), data_(data)
    {
        setResult(data);
    }

    std::shared_ptr<DataObject> InputNode::execute()
    {
        return data_;
    }

    // MultiInputTaskNode实现
    MultiInputTaskNode::MultiInputTaskNode(TaskId id, ProcessFunction func,
                                           const std::vector<std::shared_ptr<TaskNode>> &inputs)
        : TaskNode(id), func_(func), inputs_(inputs)
    {
        for (const auto &input : inputs)
        {
            if (input)
            {
                addDependency(input);
            }
        }
    }

    std::shared_ptr<DataObject> MultiInputTaskNode::execute()
    {
        if (!isReady() || getDependencies().empty())
        {
            LOG.warning("Task [%s] not ready or has no dependencies", getId().c_str());
            return nullptr;
        }

        try {
            std::vector<std::shared_ptr<DataObject>> inputResults;
            
            // 安全地获取所有依赖结果
            for (const auto &dep : getDependencies())
            {
                if (!dep) {
                    LOG.error("Null dependency in task [%s]", getId().c_str());
                    continue;
                }
                
                auto result = dep->getResult();
                inputResults.push_back(result);
            }

            // 检查是否所有输入都有效
            if (std::any_of(inputResults.begin(), inputResults.end(),
                            [](const std::shared_ptr<DataObject> &obj)
                            { return !obj; }))
            {
                LOG.warning("Some input results are null for task [%s]", getId().c_str());
                return nullptr;
            }

            // 确保处理函数存在
            if (!func_) {
                LOG.error("Process function is null for task [%s]", getId().c_str());
                return nullptr;
            }

            // 执行任务处理函数
            return func_(inputResults);
        } catch (const std::exception& e) {
            LOG.error("Exception in MultiInputTaskNode::execute: %s", e.what());
            return nullptr;
        } catch (...) {
            LOG.error("Unknown exception in MultiInputTaskNode::execute");
            return nullptr;
        }
    }

    bool MultiInputTaskNode::isReady() const
    {
        return TaskNode::isReady() && !getDependencies().empty();
    }

} // namespace GryFlux