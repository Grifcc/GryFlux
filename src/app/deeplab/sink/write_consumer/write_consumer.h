#pragma once

#include "framework/data_consumer.h"
#include "utils/logger.h"

#include <memory>
#include <filesystem>

namespace GryFlux
{
    class WriteConsumer : public DataConsumer
    {
    private:
        int processedFrames;
        std::string outputPath;

    public:
        WriteConsumer(StreamingPipeline &pipeline, std::atomic<bool> &running, CPUAllocator *allocator, std::string_view writePath = "./outputs")
            : DataConsumer(pipeline, running, allocator), processedFrames(0) {
            // 设置输出路径
            if (!writePath.empty())
            {
                // 创建目录
                std::filesystem::create_directories(writePath);
                outputPath = writePath.data();
                LOG.info("[WriteConsumer] Output path set to: %s", writePath.data());
            }
            else
            {
                LOG.error("[WriteConsumer] Invalid output path");
            }
        }

        int getProcessedFrames() const
        {
            return processedFrames;
        }

    protected:
        void run() override;
    };
}
