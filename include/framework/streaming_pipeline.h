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

#include <atomic>
#include <cstdint>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "framework/pipeline_builder.h"
#include "framework/pipeline_builder_pool.h"
#include "framework/pipeline_instance.h"
#include "utils/threadsafe_queue.h"

namespace GryFlux
{

    // 流式处理管道，用于处理持续输入的数据
    class StreamingPipeline
    {
    public:
        using ProcessorFunction = std::function<void(std::shared_ptr<PipelineBuilder>,
                                                     std::shared_ptr<DataObject>,
                                                     const std::string &)>;

        StreamingPipeline(size_t workerCount = 0,
                          size_t queueSize = 100);
        ~StreamingPipeline();

        // 启动流式处理
        void start();

        // 停止流式处理
        void stop();

        // 设置处理函数
        void setProcessor(ProcessorFunction processor);

        // 添加输入数据
        bool addInput(std::shared_ptr<DataObject> data);

        // 尝试获取输出，非阻塞
        bool tryGetOutput(std::shared_ptr<DataObject> &output);

        // 获取输出，阻塞直到有输出或流关闭
        void getOutput(std::shared_ptr<DataObject> &output);

        // 设置输出节点ID
        void setOutputNodeId(const std::string &outputId);

        // 检查输入队列是否为空
        bool inputEmpty() const;

        // 检查输出队列是否为空
        bool outputEmpty() const;

        // 获取输入队列大小
        size_t inputSize() const;

        // 获取输出队列大小
        size_t outputSize() const;

        // 获取已处理的项目数量
        size_t getProcessedItemCount() const;

        // 获取处理错误数量
        size_t getErrorCount() const;

        // 检查管道是否正在运行
        bool isRunning() const;

        // 检查输入是否活跃
        bool isInputActive() const { return input_active_.load(); }

        // 检查输出是否活跃
        bool isOutputActive() const { return output_active_.load(); }

        // 设置是否启用性能分析
        void enableProfiling(bool enable) { profilingEnabled_ = enable; }

        // 获取性能分析状态
        bool isProfilingEnabled() const { return profilingEnabled_; }

    private:
        void processingLoop(size_t workerIndex);
        void launchWorkers();
        void joinWorkers();

        // 对象池相关方法
        void initializeInstancePool();
        void clearInstancePool();
        std::shared_ptr<PipelineInstance> acquireInstance();
        void releaseInstance(const std::shared_ptr<PipelineInstance> &instance);

        using DataObjectQueue = std::shared_ptr<threadsafe_queue<std::shared_ptr<DataObject>>>;
        DataObjectQueue inputQueue_;
        std::atomic<bool> input_active_;
        DataObjectQueue outputQueue_;
        std::atomic<bool> output_active_;

        ProcessorFunction processor_;
        std::string outputNodeId_;
        std::thread processingThread_;
        std::atomic<bool> running_;
        size_t queueMaxSize_;
        size_t workerCount_;
        size_t instancePoolSize_;

        std::vector<std::thread> processingWorkers_;

        mutable std::mutex statsMutex_;

        std::vector<std::shared_ptr<PipelineInstance>> instancePool_;
        std::deque<std::shared_ptr<PipelineInstance>> instanceFreeList_;
        mutable std::mutex instanceMutex_;
        std::condition_variable instanceCv_;

        std::unique_ptr<PipelineBuilderPool> builderPool_;

        // 统计信息
        std::atomic<size_t> processedItems_;
        std::atomic<size_t> errorCount_;
        double totalProcessingTime_; // 单位：毫秒

        // 是否启用性能分析
        bool profilingEnabled_ = false;

        // 用于存储同名任务的统计数据: <任务名称, <总执行时间, 执行次数>>
        std::unordered_map<std::string, std::pair<double, size_t>> taskStats_;

        std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;
    };

} // namespace GryFlux
