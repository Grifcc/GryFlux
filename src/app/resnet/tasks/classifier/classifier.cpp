#include "classifier.h"
#include "package.h"
#include "utils/logger.h"
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

namespace GryFlux
{
    std::shared_ptr<DataObject> Classifier::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        // 参数检查
        if (inputs.size() != 2) {
            LOG.error("Classifier: inputs size is not 2");
            return nullptr;
        }

        auto preprocess_img = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
        if (!preprocess_img) {
            LOG.error("Classifier: first input is not an ImagePackage");
            return nullptr;
        }
        
        int img_id = preprocess_img->get_id();

        auto input_data = std::dynamic_pointer_cast<RunnerPackage>(inputs[1]);
        if (!input_data) {
            LOG.error("Classifier: second input is not a RunnerPackage");
            return nullptr;
        }

        if (input_data->size() == 0) {
            LOG.error("Classifier: runner output is empty");
            return nullptr;
        }

        // 获取模型输出 - ResNet输出为1x1000的概率分布
        auto [output_data, output_count] = input_data->get_output()[0];
        
        if (output_count != 1000) {
            LOG.error("Classifier: expected 1000 classes, got %zu", output_count);
            return nullptr;
        }

        LOG.info("Processing ResNet classification output with %zu classes", output_count);
        
        // 复制数据以便处理
        std::vector<float> probs(output_data.get(), output_data.get() + output_count);
        
        // 应用softmax函数（如果模型输出是logits而非概率）
        float max_val = *std::max_element(probs.begin(), probs.end());
        float sum = 0.0f;
        
        for (auto &p : probs) {
            p = std::exp(p - max_val);
            sum += p;
        }
        
        // 归一化
        if (sum > 0) {
            for (auto &p : probs) {
                p /= sum;
            }
        }
        
        // 创建索引数组并按概率排序
        std::vector<size_t> indices(output_count);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                         [&probs](size_t i1, size_t i2) { return probs[i1] > probs[i2]; });
        
        // 创建分类结果对象
        auto classification_data = std::make_shared<ClassificationPackage>(img_id);
        
        // 输出Top-5结果
        LOG.info("Top-5 Classification Results:");
        for (int i = 0; i < 5; ++i) {
            size_t idx = indices[i];
            float prob = probs[idx];
            
            // 获取类别名称
            std::string class_name;
            if (!class_labels_.empty() && idx < class_labels_.size()) {
                class_name = class_labels_[idx];
            } else {
                class_name = "class_" + std::to_string(idx);
            }
            
            // 添加到结果中
            classification_data->add_result(static_cast<int>(idx), prob, class_name);
            
            // 打印结果
            LOG.info("  Rank %d: Class %zu (%s) - Probability: %.4f", 
                    i + 1, idx, class_name.c_str(), prob * 100.0f);
        }

        // 返回分类结果
        return classification_data;
    }
} 