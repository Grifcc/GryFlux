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
#include "framework/pipeline_builder.h"
#include "utils/logger.h"
#include <chrono>
#include <iostream>
#include <unordered_map>

namespace GryFlux
{

    PipelineBuilder::PipelineBuilder(size_t numThreads) : scheduler_(std::make_shared<TaskScheduler>(numThreads)) {}

    std::shared_ptr<TaskNode> PipelineBuilder::addInput(const std::string &id, std::shared_ptr<DataObject> data)
    {
        auto inputNode = std::make_shared<InputNode>(id, data);
        scheduler_->addTask(inputNode);
        return inputNode;
    }

    std::shared_ptr<TaskNode> PipelineBuilder::addTask(
        const std::string &id,
        std::function<std::shared_ptr<DataObject>(const std::vector<std::shared_ptr<DataObject>> &)> func,
        const std::vector<std::shared_ptr<TaskNode>> &inputs)
    {

        auto node = std::make_shared<MultiInputTaskNode>(id, func, inputs);
        scheduler_->addTask(node);
        return node;
    }

    std::shared_ptr<DataObject> PipelineBuilder::execute(const std::string &outputId)
    {
        // 只有在启用性能分析时才测量时间
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        if (profilingEnabled_)
        {
            start = std::chrono::high_resolution_clock::now();
        }

        // 执行管道
        auto result = scheduler_->execute(outputId);

        // 只有在启用性能分析时才收集执行时间信息，但不输出全局统计
        if (profilingEnabled_)
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end - start).count();
            LOG.debug("Pipeline execution completed in %.3f ms", duration);

            // 仅打印当前执行的任务时间，不打印全局统计
            LOG.debug("Current execution - task times:");
            auto taskTimes = scheduler_->getTaskExecutionTimes();
            for (const auto &taskTime : taskTimes)
            {
                LOG.debug("  - Task [%s]: %.3f ms", taskTime.first.c_str(), taskTime.second);
            }
        }

        return result;
    }

    void PipelineBuilder::reset()
    {
        if (!scheduler_)
        {
            scheduler_ = std::make_shared<TaskScheduler>();
            return;
        }
        scheduler_->clear();
    }

} // namespace GryFlux