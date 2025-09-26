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

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "framework/data_object.h"
#include "framework/pipeline_builder.h"
#include "framework/pipeline_builder_pool.h"

namespace GryFlux
{
    // PipelineInstance 封装可复用的 PipelineBuilder 及其调度器
    class PipelineInstance
    {
    public:
        using ProcessorFunction = std::function<void(std::shared_ptr<PipelineBuilder>,
                                                     std::shared_ptr<DataObject>,
                                                     const std::string &)>;

        explicit PipelineInstance(PipelineBuilderPool *builderPool);

        // 准备实例以处理新的输入
        void prepare(const ProcessorFunction &processor,
                     std::shared_ptr<DataObject> input,
                     const std::string &outputNodeId,
                     bool enableProfiling);

        // 执行预构建的管线
        std::shared_ptr<DataObject> execute(const std::string &outputNodeId);

        // 获取底层 PipelineBuilder
        std::shared_ptr<PipelineBuilder> getBuilder() const { return builder_; }

        // 清理内部状态，为下次使用做准备
        void reset();

        // 获取上一次使用时间
        std::chrono::steady_clock::time_point lastUsed() const { return lastUsedTime_; }

    private:
        std::shared_ptr<PipelineBuilder> builder_;
        PipelineBuilderPool *builderPool_;
        std::chrono::steady_clock::time_point lastUsedTime_;
        bool graphInitialized_;
    };

} // namespace GryFlux
