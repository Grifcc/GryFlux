#include "image_producer.h"

#include "package.h"

namespace GryFlux {
ImageProducer::ImageProducer(StreamingPipeline &pipeline,
                             std::atomic<bool> &running,
                             CPUAllocator *allocator,
                             std::string_view dataset_path,
                             std::size_t max_frames)
    : DataProducer(pipeline, running, allocator),
      frame_count(0),
      max_frames(max_frames) {
	// read dataset path
  if (!std::filesystem::exists(dataset_path) || !std::filesystem::is_directory(dataset_path)) {
    LOG.error("Failed to open %s", dataset_path.data());
    throw std::runtime_error("wrong dataset path");
  }

  this->dataset_path_ = dataset_path;
}

void ImageProducer::run() {
  LOG.info("[ImageProducer] Producer start");
  for (auto &entry : std::filesystem::directory_iterator(dataset_path_)) {
    if (entry.is_regular_file()) {
      std::string file_path = entry.path().string();
      LOG.info("[ImageProducer] Reading file %s", file_path.c_str());
	  // read image with .jpg or .png
      if (file_path.find(".jpg") != std::string::npos ||
          file_path.find(".png") != std::string::npos) {

        cv::Mat src_frame = cv::imread(file_path);
        if (src_frame.empty()) {
          LOG.error("Failed to read image %s", file_path);
          continue;
        }

        auto input_data = std::make_shared<ImagePackage>(
            src_frame, frame_count);
        // addd data to pipeline
        if (!addData(input_data)) {
          LOG.error("[ImageProducer] Failed to add input data to pipeline");
          break;
        }
        frame_count++;
        if (frame_count >= max_frames) {
          break;
        }
      }
    }
  }
  LOG.info("[ImageProducer] Producer finished, generated %d frames",
           frame_count);
  this->stop();
}

ImageProducer::~ImageProducer() {}
};  // namespace GryFlux