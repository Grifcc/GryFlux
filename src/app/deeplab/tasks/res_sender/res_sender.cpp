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

#include "res_sender.h"
#include "package.h"
#include "utils/logger.h"

namespace GryFlux
{
    // 可选：为每个类别生成一种颜色
    static cv::Vec3b get_class_color(int class_id) {
        static std::vector<cv::Vec3b> palette = {
            {128, 64,128}, {244, 35,232}, { 70, 70, 70}, {102,102,156}, {190,153,153},
            {153,153,153}, {250,170, 30}, {220,220,  0}, {107,142, 35}, {152,251,152},
            { 70,130,180}, {220, 20, 60}, {255,  0,  0}, {  0,  0,142}, {  0,  0, 70},
            {  0, 60,100}, {  0, 80,100}, {  0,  0,230}, {119, 11, 32}, {  0,  0,  0},
            {  0,  0,  0}
        };
        return palette[class_id % palette.size()];
    }

    std::shared_ptr<DataObject> ResSender::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        // 输入为原图和分割mask
        if (inputs.size() != 2)
            return nullptr;

        auto image_data = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
        int img_id = image_data->get_id();
        auto img = image_data->get_data();

        auto object_data = std::dynamic_pointer_cast<MaskPackage>(inputs[1]);
        const cv::Mat& mask = object_data->get_mask();

        // 生成彩色分割图
        cv::Mat color_mask(mask.size(), CV_8UC3);
        for (int i = 0; i < mask.rows; ++i) {
            for (int j = 0; j < mask.cols; ++j) {
                int cls = mask.at<uchar>(i, j);
                color_mask.at<cv::Vec3b>(i, j) = get_class_color(cls);
            }
        }

        // 叠加到原图（可调alpha）
        cv::Mat img_color;
        if (img.channels() == 1)
            cv::cvtColor(img, img_color, cv::COLOR_GRAY2BGR);
        else
            img.copyTo(img_color);

        double alpha = 0.5;
        cv::addWeighted(img_color, 1 - alpha, color_mask, alpha, 0, img_color);

        return std::make_shared<MaskPackage>(img_id, img_color);
    }
}
