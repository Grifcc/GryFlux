#include "write_consumer.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <vector>
#include <thread>

#include <opencv2/imgcodecs.hpp>

#include "package.h"

namespace GryFlux
{
    WriteConsumer::WriteConsumer(StreamingPipeline &pipeline,
                                 std::atomic<bool> &running,
                                 CPUAllocator *allocator,
                                 std::string_view output_dir,
                                 std::string_view output_video_path,
                                 double video_fps)
        : DataConsumer(pipeline, running, allocator),
          processedFrames_(0),
          outputPath_(output_dir),
          videoPath_(output_video_path),
          activeVideoPath_(videoPath_),
          activeCodecDesc_(""),
          videoFps_(video_fps),
          videoEnabled_(!videoPath_.empty()),
          writerInitialized_(false)
    {
        if (!outputPath_.empty())
        {
            std::filesystem::create_directories(outputPath_);
        }

        if (videoEnabled_)
        {
            auto videoDir = std::filesystem::path(videoPath_).parent_path();
            if (!videoDir.empty())
            {
                std::filesystem::create_directories(videoDir);
            }
        }
    }

    void WriteConsumer::run()
    {
        LOG.info("[WriteConsumer] Consumer started");

        while (shouldContinue())
        {
            std::shared_ptr<DataObject> output;

            if (getData(output))
            {
                auto result = std::dynamic_pointer_cast<ImagePackage>(output);
                if (result)
                {
                    processedFrames_++;
                    auto img = result->get_data();

                    if (!outputPath_.empty())
                    {
                        auto filename = outputPath_ + "/frame_" + std::to_string(processedFrames_) + ".jpg";
                        cv::imwrite(filename, img);
                        LOG.info("[WriteConsumer] Frame %d saved to %s", processedFrames_, filename.c_str());
                    }

                    if (videoEnabled_)
                    {
                        if (!writerInitialized_)
                        {
                            if (!initializeWriter(img))
                            {
                                LOG.error("[WriteConsumer] Failed to initialize video writer at %s", videoPath_.c_str());
                                videoEnabled_ = false;
                            }
                            else
                            {
                                writerInitialized_ = true;
                                if (activeVideoPath_ != videoPath_)
                                {
                                    LOG.warning("[WriteConsumer] Requested video path %s replaced with %s to match codec",
                                                videoPath_.c_str(),
                                                activeVideoPath_.c_str());
                                }
                                LOG.info("[WriteConsumer] Video writing enabled at %s using %s",
                                         activeVideoPath_.c_str(),
                                         activeCodecDesc_.empty() ? "unknown" : activeCodecDesc_.c_str());
                            }
                        }

                        if (writerInitialized_)
                        {
                            writer_.write(img);
                        }
                    }
                }
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }

        LOG.info("[WriteConsumer] Processed frames: %d", processedFrames_);

        if (writerInitialized_)
        {
            writer_.release();
            LOG.info("[WriteConsumer] Video writer released (file saved at %s)", activeVideoPath_.c_str());
        }
    }
}

namespace GryFlux
{
    bool WriteConsumer::initializeWriter(const cv::Mat &sampleFrame)
    {
        if (sampleFrame.empty())
        {
            LOG.error("[WriteConsumer] Cannot create video writer with empty frame");
            return false;
        }

        if (sampleFrame.channels() != 1 && sampleFrame.channels() != 3)
        {
            LOG.warning("[WriteConsumer] Unexpected channel count %d; attempting to treat frame as %s",
                         sampleFrame.channels(),
                         sampleFrame.channels() == 1 ? "grayscale" : "color");
        }

        struct CodecCandidate
        {
            int fourcc;
            std::string codecName;
            std::string extension;
        };

        std::vector<CodecCandidate> candidates;
        std::string extension = std::filesystem::path(videoPath_).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        const bool prefersMp4 = extension.empty() || extension == ".mp4";

        auto addCandidate = [&](int fourcc, std::string name, std::string ext = std::string()) {
            candidates.push_back({fourcc, std::move(name), std::move(ext)});
        };

        if (prefersMp4)
        {
            addCandidate(cv::VideoWriter::fourcc('a', 'v', 'c', '1'), "H.264 (avc1)");
            addCandidate(cv::VideoWriter::fourcc('h', '2', '6', '4'), "H.264 (H264)");
            addCandidate(cv::VideoWriter::fourcc('m', 'p', '4', 'v'), "MPEG-4 Part 2");
        }

        addCandidate(cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), "XVID MPEG-4", ".avi");
        addCandidate(cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), "Motion JPEG", ".avi");

        for (const auto &candidate : candidates)
        {
            if (openWriterWithCodec(sampleFrame, candidate.fourcc, candidate.codecName, candidate.extension))
            {
                return true;
            }
        }

        return false;
    }

    bool WriteConsumer::openWriterWithCodec(const cv::Mat &sampleFrame,
                                            int fourcc,
                                            const std::string &codecName,
                                            const std::string &extensionOverride)
    {
        auto targetPath = std::filesystem::path(videoPath_);

        if (!extensionOverride.empty())
        {
            if (targetPath.empty() || targetPath.filename().empty())
            {
                std::string fallbackName = "result" + extensionOverride;
                targetPath = std::filesystem::path(outputPath_.empty() ? "./" : outputPath_) / fallbackName;
            }
            else
            {
                targetPath.replace_extension(extensionOverride);
            }
        }

        if (!targetPath.parent_path().empty())
        {
            std::filesystem::create_directories(targetPath.parent_path());
        }

        const bool isColor = sampleFrame.channels() != 1;

        if (!writer_.open(targetPath.string(), fourcc, videoFps_, sampleFrame.size(), isColor))
        {
            LOG.warning("[WriteConsumer] Video writer open failed using %s at %s",
                         codecName.c_str(),
                         targetPath.string().c_str());
            return false;
        }

        activeVideoPath_ = targetPath.string();
        activeCodecDesc_ = codecName;
        return true;
    }
}
