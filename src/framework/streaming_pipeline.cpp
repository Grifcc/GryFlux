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
#include "framework/streaming_pipeline.h"
#include "utils/logger.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <cmath>

namespace GryFlux
{

    StreamingPipeline::StreamingPipeline(size_t workerCount, size_t schedulerThreadCount, size_t queueSize)
        : inputQueue_(std::make_shared<threadsafe_queue<std::shared_ptr<DataObject>>>()),
          outputQueue_(std::make_shared<threadsafe_queue<std::shared_ptr<DataObject>>>()),
        outputNodeId_("output"),
          running_(false),
          queueMaxSize_(queueSize),
          workerCount_(workerCount > 0 ? workerCount : 1),
          schedulerThreadCount_(schedulerThreadCount),
          processedItems_(0),
          errorCount_(0),
          totalProcessingTime_(0.0),
          profilingEnabled_(false) {}

    StreamingPipeline::~StreamingPipeline()
    {
        stop();
    }

    void StreamingPipeline::start()
    {
        if (running_.load())
        {
            return;
        }

        if (!processor_)
        {
            throw std::runtime_error("Processor function not set");
        }

        // 重置统计数据
        processedItems_.store(0);
        errorCount_.store(0);
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            taskStats_.clear();
        }

        startTime_ = std::chrono::high_resolution_clock::now();

        running_ = true;
        input_active_ = true;
        output_active_ = true;
 
        size_t instanceCount = workerCount_ > 0 ? workerCount_ : 1;
        if (instanceCount == 0)
        {
            instanceCount = 1;
        }

        workerCount_ = instanceCount;

        builderPool_ = std::make_unique<PipelineBuilderPool>(instanceCount, schedulerThreadCount_);
        initializeInstancePool();
        launchWorkers();

        LOG.debug("[Pipeline] Started streaming pipeline with %zu workers", workerCount_);
    }

    void StreamingPipeline::stop()
    {
        if (!running_.exchange(false))
        {
            return;
        }

        input_active_ = false;

        instanceCv_.notify_all();
        joinWorkers();

        output_active_ = false;

        clearInstancePool();

        if (builderPool_)
        {
            builderPool_->shutdown();
            builderPool_.reset();
        }

        // 只有在启用性能分析时才输出统计数据
        if (profilingEnabled_)
        {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto totalTime = std::chrono::duration<double, std::milli>(endTime - startTime_).count();

            LOG.info("[Pipeline] Statistics:");
            LOG.info("  - Total items processed: %zu", processedItems_);
            LOG.info("  - Error count: %zu", errorCount_);
            LOG.info("  - Total running time: %.3f ms", totalTime);

            if (processedItems_ > 0)
            {
                double avgTime = static_cast<double>(totalProcessingTime_) / processedItems_;
                LOG.info("  - Average processing time per item: %.3f ms", avgTime);
                LOG.info("  - Processing rate: %.2f items/s", (processedItems_ * 1000.0 / totalTime));
            }

            // 输出同名任务的全局平均执行时间
            if (!taskStats_.empty())
            {
                LOG.info("[Pipeline] Global average execution time for tasks with the same name:");
                for (const auto &taskStat : taskStats_)
                {
                    const std::string &taskName = taskStat.first;
                    double totalTime = taskStat.second.first;
                    size_t count = taskStat.second.second;
                    double avgTime = totalTime / count;

                    LOG.info("  - Task [%s]: %.3f ms (average of %zu executions across all items)",
                             taskName.c_str(), avgTime, count);
                }
            }
        }
        else
        {
            LOG.debug("[Pipeline] Stopped streaming pipeline");
        }
    }

    void StreamingPipeline::setProcessor(ProcessorFunction processor)
    {
        if (running_.load())
        {
            throw std::runtime_error("Cannot set processor while pipeline is running");
        }
        processor_ = std::move(processor);
    }

    bool StreamingPipeline::addInput(std::shared_ptr<DataObject> data)
    {
        if (!data || !input_active_.load())
        {
            return false;
        }

        if (queueMaxSize_ > 0)
        {
            while (input_active_.load() && static_cast<size_t>(inputQueue_->size()) >= queueMaxSize_)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        if (!input_active_.load())
        {
            return false;
        }

        inputQueue_->push(std::move(data));
        return true;
    }

    bool StreamingPipeline::tryGetOutput(std::shared_ptr<DataObject> &output)
    {
        return outputQueue_->try_pop(output);
    }

    void StreamingPipeline::getOutput(std::shared_ptr<DataObject> &output)
    {
        outputQueue_->wait_and_pop(output);
    }

    void StreamingPipeline::setOutputNodeId(const std::string &outputId)
    {
        if (running_.load())
        {
            throw std::runtime_error("Cannot set output node ID while pipeline is running");
        }
        outputNodeId_ = outputId;
    }

    bool StreamingPipeline::inputEmpty() const
    {
        return inputQueue_->empty();
    }

    bool StreamingPipeline::outputEmpty() const
    {
        return outputQueue_->empty();
    }

    size_t StreamingPipeline::inputSize() const
    {
        return static_cast<size_t>(inputQueue_->size());
    }

    size_t StreamingPipeline::outputSize() const
    {
        return static_cast<size_t>(outputQueue_->size());
    }

    size_t StreamingPipeline::getProcessedItemCount() const
    {
        return processedItems_.load();
    }
    
    size_t StreamingPipeline::getErrorCount() const
    {
        return errorCount_.load();
    }

    bool StreamingPipeline::isRunning() const
    {
        return running_.load();
    }

    void StreamingPipeline::launchWorkers()
    {
        processingWorkers_.clear();
        processingWorkers_.reserve(workerCount_);
        for (size_t i = 0; i < workerCount_; ++i)
        {
            processingWorkers_.emplace_back(&StreamingPipeline::processingLoop, this, i);
        }
    }

    void StreamingPipeline::joinWorkers()
    {
        for (auto &worker : processingWorkers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        processingWorkers_.clear();
    }

    void StreamingPipeline::initializeInstancePool()
    {
        std::lock_guard<std::mutex> lock(instanceMutex_);
        instancePool_.clear();
        instanceFreeList_.clear();
        size_t count = workerCount_ > 0 ? workerCount_ : 1;
        if (count == 0)
        {
            count = 1;
        }
        instancePool_.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            auto instance = std::make_shared<PipelineInstance>(builderPool_.get());
            instancePool_.push_back(instance);
            instanceFreeList_.push_back(std::move(instance));
        }
    }

    void StreamingPipeline::clearInstancePool()
    {
        std::lock_guard<std::mutex> lock(instanceMutex_);
        for (auto &instance : instancePool_)
        {
            if (instance)
            {
                instance->reset();
            }
        }
        instancePool_.clear();
        instanceFreeList_.clear();
    }

    std::shared_ptr<PipelineInstance> StreamingPipeline::acquireInstance()
    {
        std::unique_lock<std::mutex> lock(instanceMutex_);
        instanceCv_.wait(lock, [this]
                         { return !instanceFreeList_.empty() || !running_.load(); });

        if (instanceFreeList_.empty())
        {
            return nullptr;
        }

        auto instance = instanceFreeList_.front();
        instanceFreeList_.pop_front();
        return instance;
    }

    void StreamingPipeline::releaseInstance(const std::shared_ptr<PipelineInstance> &instance)
    {
        if (!instance)
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(instanceMutex_);
            instanceFreeList_.push_back(instance);
        }
        instanceCv_.notify_one();
    }

    void StreamingPipeline::setSchedulerThreadCount(size_t threadCount)
    {
        if (running_.load())
        {
            throw std::runtime_error("Cannot change scheduler thread count while running");
        }
        schedulerThreadCount_ = threadCount;
    }

    void StreamingPipeline::processingLoop(size_t workerIndex)
    {
        while (running_.load() || !inputQueue_->empty())
        {
            std::shared_ptr<DataObject> input;
            if (!inputQueue_->try_pop(input))
            {
                if (!running_.load())
                {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            if (!input)
            {
                continue;
            }

            auto instance = acquireInstance();
            if (!instance)
            {
                break;
            }

            std::chrono::time_point<std::chrono::high_resolution_clock> frameStart;
            if (profilingEnabled_)
            {
                frameStart = std::chrono::high_resolution_clock::now();
            }

            std::unordered_map<std::string, double> frameTaskTimes;
            try
            {
                instance->prepare(processor_, input, outputNodeId_, profilingEnabled_);

                auto result = instance->execute(outputNodeId_);
                if (result)
                {
                    outputQueue_->push(std::move(result));
                    processedItems_.fetch_add(1, std::memory_order_relaxed);
                }
                else
                {
                    LOG.error("[Pipeline] Pipeline execution returned null output");
                    errorCount_.fetch_add(1, std::memory_order_relaxed);
                }

                if (profilingEnabled_)
                {
                    auto scheduler = instance->getBuilder()->getScheduler();
                    if (scheduler)
                    {
                        frameTaskTimes = scheduler->getTaskExecutionTimes();
                    }
                }
            }
            catch (const std::exception &ex)
            {
                LOG.error("[Pipeline] Execution error: %s", ex.what());
                errorCount_.fetch_add(1, std::memory_order_relaxed);
            }
            catch (...)
            {
                LOG.error("[Pipeline] Execution error: unknown exception");
                errorCount_.fetch_add(1, std::memory_order_relaxed);
            }

            if (profilingEnabled_)
            {
                const auto frameEnd = std::chrono::high_resolution_clock::now();
                const double frameDurationMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

                {
                    std::lock_guard<std::mutex> lock(statsMutex_);
                    totalProcessingTime_ += frameDurationMs;
                    for (const auto &entry : frameTaskTimes)
                    {
                        auto &stat = taskStats_[entry.first];
                        stat.first += entry.second;
                        stat.second += 1;
                    }
                }

                LOG.debug("[Pipeline] Processed item %zu in %.3f ms",
                          processedItems_.load(std::memory_order_relaxed), frameDurationMs);
            }

            instance->reset();
            releaseInstance(instance);
        }

        LOG.debug("[Pipeline] Processing loop completed");
    }

} // namespace GryFlux
