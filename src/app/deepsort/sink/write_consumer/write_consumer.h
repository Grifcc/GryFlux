#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "framework/data_consumer.h"
#include "utils/logger.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

namespace GryFlux
{
    class WriteConsumer : public DataConsumer
    {
    public:
        WriteConsumer(StreamingPipeline &pipeline,
                      std::atomic<bool> &running,
                      CPUAllocator *allocator,
                      std::string_view output_dir = "./outputs",
                      std::string_view output_video_path = "",
                      double video_fps = 25.0);

        int getProcessedFrames() const { return processedFrames_; }

    protected:
        void run() override;

    private:
        bool initializeWriter(const cv::Mat &sampleFrame);
        bool openWriterWithCodec(const cv::Mat &sampleFrame,
                                 int fourcc,
                                 const std::string &codecName,
                                 const std::string &extensionOverride = "");

        int processedFrames_;
        std::string outputPath_;
        std::string videoPath_;
        std::string activeVideoPath_;
        std::string activeCodecDesc_;
        double videoFps_;
        bool videoEnabled_;
        bool writerInitialized_;
        cv::VideoWriter writer_;
    };
}
