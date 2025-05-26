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
#include <fstream>
#include <vector>

#include "framework/streaming_pipeline.h"
#include "framework/data_object.h"
#include "framework/processing_task.h"

#include "utils/logger.h"

#include "source/producer/image_producer.h"
#include "tasks/image_preprocess/image_preprocess.h"
#include "tasks/object_detector/classifier.h"
#include "tasks/rk_runner/rk_runner.h"
#include "tasks/res_sender/res_sender.h"
#include "sink/write_consumer/write_consumer.h"

// 全局变量存储类别标签
std::vector<std::string> g_class_labels;

// 读取synset.txt文件
bool loadSynsetLabels(const std::string& synset_path) {
    std::ifstream file(synset_path);
    if (!file.is_open()) {
        std::cerr << "无法打开标签文件: " << synset_path << std::endl;
        return false;
    }
    
    g_class_labels.clear();
    std::string line;
    while (std::getline(file, line)) {
        // 解析每行，格式为 "nXXXXXXXX label_name"
        size_t pos = line.find(' ');
        if (pos != std::string::npos) {
            // 提取标签名称（空格后面的部分）
            std::string label = line.substr(pos + 1);
            g_class_labels.push_back(label);
        } else {
            // 如果没有空格，使用整行作为标签
            g_class_labels.push_back(line);
        }
    }
    
    LOG.info("成功加载 %zu 个类别标签", g_class_labels.size());
    return true;
}

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
    auto classifierNode = builder->addTask("classifier",
                                           taskRegistry.getProcessFunction("classifier"),
                                           {imgPreprocessNode, rkRunnerNode});

    builder->addTask(outputId,
                     taskRegistry.getProcessFunction("resultSender"),
                     {inputNode, classifierNode});

}

void initLogger()
{
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::BOTH);
    LOG.setAppName("ResNetClassification");
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
    if (argc != 4) {
        std::cerr << "Usage: " + std::string(argv[0]) + " <model_path> <image_path> <synset_path>" << std::endl;
        std::cerr << "  model_path: ResNet模型文件路径" << std::endl;
        std::cerr << "  image_path: 输入图像或视频路径" << std::endl;
        std::cerr << "  synset_path: 类别标签文件路径 (synset.txt)" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* image_path = argv[2];
    const char* synset_path = argv[3];
    
    initLogger();
    
    // 加载类别标签
    if (!loadSynsetLabels(synset_path)) {
        LOG.error("加载类别标签失败，使用默认标签");
    }

    // 创建全局任务注册表
    GryFlux::TaskRegistry taskRegistry;

    CPUAllocator *cpuAllocator = new CPUAllocator();
    // 注册各种处理任务
    // ResNet通常使用224x224的输入尺寸
    taskRegistry.registerTask<GryFlux::ImagePreprocess>("imagePreprocess", 224, 224);
    taskRegistry.registerTask<GryFlux::RkRunner>("rkRunner", model_path);
    // 使用新的Classifier类，并传递类别标签
    taskRegistry.registerTask<GryFlux::Classifier>("classifier", 0.0f, g_class_labels);
    taskRegistry.registerTask<GryFlux::ResSender>("resultSender");
    // 创建流式处理管道
    GryFlux::StreamingPipeline pipeline(10); // 使用10个线程
    // 设置输出节点ID
    pipeline.setOutputNodeId("resultSender");

    // 启用性能分析
    pipeline.enableProfiling(true);

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
    GryFlux::ImageProducer producer(pipeline, running, cpuAllocator, image_path);
    GryFlux::WriteConsumer consumer(pipeline, running, cpuAllocator);

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
    
    delete cpuAllocator;
    return 0;
}

