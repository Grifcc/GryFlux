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

#include <memory>
#include <string_view>

 #include "framework/data_producer.h"
 #include "utils/logger.h"
 #include "utils/unified_allocator.h"
 #include "opencv2/opencv.hpp"
 
 namespace GryFlux
 {
	 class VideoProducer : public DataProducer
	 { 
		public:
		explicit VideoProducer(StreamingPipeline &pipeline, std::atomic<bool> &running, CPUAllocator *allocator,
						   const std::string video_path, int max_frames = 100, int frame_interval_ms = 33);
		~VideoProducer();
		private:
			int frame_count;
			int max_frames;
			int frame_interval_ms;
			int width_;
			int height_;
			int channel_;
			cv::VideoCapture video_;
 
		 protected:
		 void run() override;
	 };
 }
 