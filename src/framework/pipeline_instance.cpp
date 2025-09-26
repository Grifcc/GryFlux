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
#include "framework/pipeline_instance.h"

#include <stdexcept>
#include <utility>

namespace GryFlux
{
    PipelineInstance::PipelineInstance(PipelineBuilderPool *builderPool)
        : builderPool_(builderPool),
          lastUsedTime_(std::chrono::steady_clock::now()),
          graphInitialized_(false)
    {
    }

    void PipelineInstance::prepare(const ProcessorFunction &processor,
                                   std::shared_ptr<DataObject> input,
                                   const std::string &outputNodeId,
                                   bool enableProfiling)
    {
        if (!builderPool_)
        {
            throw std::runtime_error("PipelineBuilderPool not configured");
        }

        if (!builder_)
        {
            builder_ = builderPool_->acquire();
            graphInitialized_ = false;
        }

        if (!builder_)
        {
            throw std::runtime_error("Failed to acquire PipelineBuilder from pool");
        }

        builder_->reset();
        builder_->enableProfiling(enableProfiling);
        if (processor)
        {
            processor(builder_, std::move(input), outputNodeId);
        }
        graphInitialized_ = true;
        lastUsedTime_ = std::chrono::steady_clock::now();
    }

    std::shared_ptr<DataObject> PipelineInstance::execute(const std::string &outputNodeId)
    {
        auto result = builder_->execute(outputNodeId);
        lastUsedTime_ = std::chrono::steady_clock::now();
        return result;
    }

    void PipelineInstance::reset()
    {
        if (builder_)
        {
            builder_->reset();
            builder_->enableProfiling(false);
            builder_.reset();
        }
        graphInitialized_ = false;
    }

} // namespace GryFlux
