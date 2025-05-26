#include "video_producer.h"
#include "package.h"

namespace GryFlux {
    VideoProducer::VideoProducer(StreamingPipeline &pipeline, std::atomic<bool> &running, CPUAllocator *allocator,
                            const std::string video_path, int max_frames, int frame_interval_ms)
                : DataProducer(pipeline, running, allocator), frame_count(0), max_frames(max_frames), frame_interval_ms(frame_interval_ms) {
                    this->video_ = cv::VideoCapture(video_path, cv::CAP_FFMPEG);
                    if (!(this->video_).isOpened()) {
                        LOG.error("Failed to open %s", video_path);
                        throw std::runtime_error("fail fopen");
                    }
                    this->width_ = video_.get(cv::CAP_PROP_FRAME_WIDTH);
                    this->height_ = video_.get(cv::CAP_PROP_FRAME_HEIGHT);
                }
            
	void VideoProducer::run() {
        LOG.info("[VideoProducer] Producer start");

        for (int i = 0; i < max_frames && running.load(); ++i)
        {
            // 生成测试图像
            cv::Mat src_frame;
			if (!video_.read(src_frame)) {
				LOG.info("Failed to read frame");
				continue;
			}
            auto input_data = std::make_shared<ImagePackage>(
                src_frame, i
            );
            // 添加到处理管道
            if (!addData(input_data))
            {
                LOG.error("[VideoProducer] Failed to add input data to pipeline");
                break;
            }

            frame_count++;

            // 模拟帧率
            // std::this_thread::sleep_for(std::chrono::milliseconds(frame_interval_ms));
        }
        this->stop();
        LOG.info("[VideoProducer] Producer finished, generated %d frames", frame_count);
    }
    VideoProducer::~VideoProducer() {
        video_.release();
    }

};