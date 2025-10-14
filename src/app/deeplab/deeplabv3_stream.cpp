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
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <memory>
#include <filesystem>
#include <functional>
#include <unordered_map>

#include "framework/streaming_pipeline.h"
#include "framework/data_object.h"
#include "framework/processing_task.h"

#include "utils/logger.h"

#include "source/producer/image_producer.h"
#include "tasks/image_preprocess/image_preprocess.h"
#include "tasks/object_detector/object_detector.h"
#include "tasks/rk_runner/rk_runner.h"
#include "tasks/res_sender/res_sender.h"
#include "sink/write_consumer/write_consumer.h"
// 计算图构建函数
void buildStreamingComputeGraph(std::shared_ptr<GryFlux::PipelineBuilder> builder,
                                std::shared_ptr<GryFlux::DataObject> input,
                                const std::string &outputId,
                                GryFlux::TaskRegistry &taskRegistry)
{
    // 输入节点
    auto inputNode = builder->addInput("input", input);

    // 使用注册表中的任务构建计算图
    auto imgPreprocessNode = builder->addTask("imagePreprocess",
                                              taskRegistry.getProcessFunction("imagePreprocess"),
                                              {inputNode});
    auto rkRunnerNode = builder->addTask("rkRunner",
                                         taskRegistry.getProcessFunction("rkRunner"),
                                         {imgPreprocessNode});
    auto objectDetectorNode = builder->addTask("objectDetector",
                                               taskRegistry.getProcessFunction("objectDetector"),
                                               {imgPreprocessNode, rkRunnerNode});

    builder->addTask(outputId,
                     taskRegistry.getProcessFunction("resultSender"),
                     {imgPreprocessNode, objectDetectorNode});

}

void initLogger()
{
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::BOTH);
    LOG.setAppName("StreamingExample");
    //  如果logs目录不存在，创建logs目录
    std::filesystem::path dirPath("./logs");
    if (!std::filesystem::exists(dirPath))
    {
        try
        {
            std::filesystem::create_directories(dirPath);
        }
        catch (const std::exception &e)
        {
            LOG.error("无法创建日志目录: %s", e.what());
        }
    }
    LOG.setLogFileRoot("./logs");
}

int main(int argc, const char **argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " + std::string(argv[0]) + "<model_path>  <dataset_path>" << std::endl;
        return 1;
    }
    initLogger();

    // 创建全局任务注册表
    GryFlux::TaskRegistry taskRegistry;

    CPUAllocator *cpuAllocator = new CPUAllocator();
    // 注册各种处理任务
    taskRegistry.registerTask<GryFlux::ImagePreprocess>("imagePreprocess", 513, 513);
    taskRegistry.registerTask<GryFlux::RkRunner>("rkRunner", argv[1]);
    taskRegistry.registerTask<GryFlux::ObjectDetector>("objectDetector", 0.5f);
    taskRegistry.registerTask<GryFlux::ResSender>("resultSender");
    // 创建流式处理管道
    GryFlux::StreamingPipeline pipeline(10); // 使用10个线程
    // 设置输出节点ID
    pipeline.setOutputNodeId("resultSender");

    // 启用性能分析
    pipeline.enableProfiling(true);

    // 设置输出节点ID

    // 设置处理函数
    pipeline.setProcessor([&taskRegistry](std::shared_ptr<GryFlux::PipelineBuilder> builder,
                                          std::shared_ptr<GryFlux::DataObject> input,
                                          const std::string &outputId)
                          {
        // 调用命名函数
        buildStreamingComputeGraph(builder, input, outputId, taskRegistry); });

    // 启动管道
    pipeline.start();

    // 创建控制标志，表示是否仍在运行
    std::atomic<bool> running(true);

    // 创建输入生产者和消费者
    GryFlux::ImageProducer producer(pipeline, running, cpuAllocator, argv[2]);
    GryFlux::WriteConsumer consumer(pipeline, running,cpuAllocator);

    // 启动生产者和消费者
    producer.start();
    consumer.start();

    // 等待生产者和消费者线程结束
    producer.join();
    LOG.info("[main] Producer finished");

    consumer.join();
    LOG.info("[main] Consumer finished, processed %d frames", consumer.getProcessedFrames());

    pipeline.stop();
    LOG.info("[main] Pipeline stopped");
    return 0;
}

