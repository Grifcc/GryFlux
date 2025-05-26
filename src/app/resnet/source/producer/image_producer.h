 #pragma once

#include <memory>
#include <string_view>
#include <filesystem>

#include "framework/data_producer.h"
#include "utils/logger.h"
#include "utils/unified_allocator.h"
#include "opencv2/opencv.hpp"
 
 namespace GryFlux
 {
	 class ImageProducer : public DataProducer
	 { 
		public:
		explicit ImageProducer(StreamingPipeline &pipeline, std::atomic<bool> &running, CPUAllocator *allocator,
						   std::string_view dataset_path, std::size_t max_frames = -1);
		~ImageProducer();
		private:
			int frame_count;
			std::size_t max_frames;
			std::filesystem::path dataset_path_;
 
		 protected:
		 void run() override;
	 };
 }
