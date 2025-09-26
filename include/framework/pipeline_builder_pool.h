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
#include <memory>
#include <mutex>
#include <vector>

#include "framework/pipeline_builder.h"

namespace GryFlux
{

    class PipelineBuilderPool
    {
    public:
        explicit PipelineBuilderPool(size_t capacity);
        ~PipelineBuilderPool();

        PipelineBuilderPool(const PipelineBuilderPool &) = delete;
        PipelineBuilderPool &operator=(const PipelineBuilderPool &) = delete;

        std::shared_ptr<PipelineBuilder> acquire();
        void shutdown();

    private:
        void release(const std::shared_ptr<PipelineBuilder> &builder);

        size_t capacity_;

        std::mutex mutex_;
        std::condition_variable cv_;
        std::vector<std::shared_ptr<PipelineBuilder>> allBuilders_;
        std::deque<std::shared_ptr<PipelineBuilder>> idleBuilders_;
        bool stopped_;
    };

} // namespace GryFlux
