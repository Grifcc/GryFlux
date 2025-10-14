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
#include <sstream>
#include <iomanip>

namespace GryFlux
{
    // ImageNet 1000类别名称，可根据实际使用的模型替换
    static const std::vector<std::string> IMAGENET_CLASSES = {
        "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead", "electric_ray", "stingray", "cock", "hen", "ostrich", 
        "brambling", "goldfinch", "house_finch", "junco", "indigo_bunting", "robin", "bulbul", "jay", "magpie", "chickadee", 
        "water_ouzel", "kite", "bald_eagle", "vulture", "great_grey_owl", "European_fire_salamander", "common_newt", "eft", "spotted_salamander", "axolotl", 
        "bullfrog", "tree_frog", "tailed_frog", "loggerhead", "leatherback_turtle", "mud_turtle", "terrapin", "box_turtle", "banded_gecko", "common_iguana", 
        "American_chameleon", "whiptail", "agama", "frilled_lizard", "alligator_lizard", "Gila_monster", "green_lizard", "African_chameleon", "Komodo_dragon", "African_crocodile", 
        // 这里省略了大部分类别，实际使用时应包含全部1000个类别名称
        "...其他类别...",
        "bolete", "ear", "toilet_tissue"
    };

    std::shared_ptr<DataObject> ResSender::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        // 输入图像和分类结果
        if (inputs.size() != 2) {
            LOG.error("ResSender: inputs size is not 2");
            return nullptr;
        }

        auto image_data = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
        if (!image_data) {
            LOG.error("ResSender: first input is not an ImagePackage");
            return nullptr;
        }
        
        int img_id = image_data->get_id();
        auto img = image_data->get_data();

        // 获取ClassificationPackage
        auto classification_data = std::dynamic_pointer_cast<ClassificationPackage>(inputs[1]);
        if (!classification_data) {
            LOG.error("ResSender: second input is not a ClassificationPackage");
            return nullptr;
        }
        
        // 使用ClassificationPackage处理
        auto results = classification_data->get_results();
        
        // 绘制分类结果
        int y_offset = 30;
        cv::putText(img, "ResNet Classification Results:", cv::Point(10, y_offset), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        y_offset += 30;
        
        for (size_t i = 0; i < results.size(); ++i) {
            auto result = results[i];
            
            // 格式化概率值为百分比
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << (result.probability * 100.0);
            
            // 构建显示文本
            std::string display_text = std::to_string(i + 1) + ". " + 
                                      result.class_name + ": " + ss.str() + "%";
            
            // 根据排名设置不同颜色
            cv::Scalar text_color;
            if (i == 0) {
                text_color = cv::Scalar(0, 0, 255); // 第一名红色
            } else if (i == 1) {
                text_color = cv::Scalar(0, 165, 255); // 第二名橙色
            } else if (i == 2) {
                text_color = cv::Scalar(0, 255, 255); // 第三名黄色
            } else {
                text_color = cv::Scalar(255, 0, 0); // 其他蓝色
            }
            
            // 在图像上绘制文本
            cv::putText(img, display_text, cv::Point(10, y_offset), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
            y_offset += 30;
        }
        
        // 添加分割线
        cv::line(img, cv::Point(10, img.rows - 30), cv::Point(img.cols - 10, img.rows - 30), 
                cv::Scalar(200, 200, 200), 1);
        
        return std::make_shared<ImagePackage>(img, img_id);
    }
}
