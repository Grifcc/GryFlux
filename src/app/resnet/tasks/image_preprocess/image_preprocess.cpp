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
#include "image_preprocess.h"
#include "package.h"
#include "utils/logger.h"

namespace GryFlux
{
    std::shared_ptr<DataObject> ImagePreprocess::process(const std::vector<std::shared_ptr<DataObject>>& inputs) {
        // Check input validity
        if (inputs.size() != 1) {
            return nullptr;
        }

        auto input_data = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
        const auto frame = input_data->get_data();
        cv::Mat img;
        
        // Convert BGR to RGB (OpenCV default is BGR, models typically expect RGB)
        cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);

        // Get original image dimensions and ID
        int img_width = input_data->get_width();
        int img_height = input_data->get_height();
        int idx = input_data->get_id();

        // Early return if image already matches model input size
        if (img_width == model_width_ && img_height == model_height_) {
            return std::make_shared<ImagePackage>(img, idx, 1.0f, 0, 0);
        }

        // --- Letterbox Preprocessing ---
        // 1. Calculate scaling factor (maintain aspect ratio)
        float scale = std::min(static_cast<float>(model_width_) / img_width,
                            static_cast<float>(model_height_) / img_height);
        
        // 2. Calculate new dimensions after scaling
        int new_w = static_cast<int>(img_width * scale);
        int new_h = static_cast<int>(img_height * scale);
        
        // 3. Perform aspect ratio-preserving resize
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(new_w, new_h));

        // 4. Create gray canvas (114,114,114 is YOLO's conventional padding color)
        cv::Mat letterbox_img(model_height_, model_width_, CV_8UC3, cv::Scalar(114, 114, 114));
        
        // 5. Calculate centering offsets
        int x_offset = (model_width_ - new_w) / 2;  // Horizontal padding
        int y_offset = (model_height_ - new_h) / 2; // Vertical padding
        
        // 6. Place resized image in center of canvas
        resized_img.copyTo(letterbox_img(cv::Rect(x_offset, y_offset, new_w, new_h)));

        LOG.info("Letterbox processing: id=%d | Original(%dx%d) -> Scaled(%dx%d) -> Padded(%dx%d) | scale=%f, pad=(%d,%d)",
                idx, img_width, img_height, new_w, new_h, model_width_, model_height_, 
                scale, x_offset, y_offset);

        // Return processed package with transformation metadata
        return std::make_shared<ImagePackage>(
            letterbox_img,    // Processed image with letterbox
            idx,              // Original image ID
            scale,            // Scaling factor (same for width/height due to aspect ratio preservation)
            x_offset,         // Horizontal padding (left side)
            y_offset          // Vertical padding (top side)
        );
    }
}
