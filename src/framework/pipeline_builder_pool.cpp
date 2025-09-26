/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicence, and/or sell copies of the Software, and to
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
#include "framework/pipeline_builder_pool.h"

#include <stdexcept>

namespace GryFlux
{
    PipelineBuilderPool::PipelineBuilderPool(size_t capacity)
        : capacity_(capacity == 0 ? 1 : capacity),
          stopped_(false)
    {
    }

    PipelineBuilderPool::~PipelineBuilderPool()
    {
        shutdown();
    }

    std::shared_ptr<PipelineBuilder> PipelineBuilderPool::acquire()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]
                 { return stopped_ || !idleBuilders_.empty() || allBuilders_.size() < capacity_; });

        if (stopped_)
        {
            return nullptr;
        }

        if (idleBuilders_.empty() && allBuilders_.size() < capacity_)
        {
            auto builder = std::make_shared<PipelineBuilder>();
            allBuilders_.push_back(builder);
            idleBuilders_.push_back(builder);
        }

        if (idleBuilders_.empty())
        {
            return nullptr;
        }

        auto builder = idleBuilders_.front();
        idleBuilders_.pop_front();

        return std::shared_ptr<PipelineBuilder>(builder.get(), [this, builder](PipelineBuilder *ptr)
                                                { (void)ptr; this->release(builder); });
    }

    void PipelineBuilderPool::release(const std::shared_ptr<PipelineBuilder> &builder)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_)
        {
            return;
        }
        idleBuilders_.push_back(builder);
        cv_.notify_one();
    }

    void PipelineBuilderPool::shutdown()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        idleBuilders_.clear();
        allBuilders_.clear();
        cv_.notify_all();
    }

} // namespace GryFlux
