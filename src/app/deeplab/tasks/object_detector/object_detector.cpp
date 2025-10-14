#include "object_detector.h"
#include "package.h"
#include "utils/logger.h"
namespace GryFlux
{
    const int OBJ_CLASS_NUM = 21;
    inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

    std::shared_ptr<DataObject> ObjectDetector::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        // runner and preprocess
        if (inputs.size() != 2) {
            LOG.error("ObjectDetector: inputs size is not 2");
            return nullptr;
        }

        auto preprocess_img = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
        int img_id = preprocess_img->get_id();
        float scale = preprocess_img->get_scale();
        int x_pad = preprocess_img->get_x_pad();
        int y_pad = preprocess_img->get_y_pad();
        int model_in_w = preprocess_img->get_width();
        int model_in_h = preprocess_img->get_height();

        LOG.info("Image preprocess scale: %f, x_pad: %d, y_pad: %d", scale, x_pad, y_pad);

        auto input_data = std::dynamic_pointer_cast<RunnerPackage>(inputs[1]);
        auto [output_data, output_cnt] = input_data->get_output()[0];
        LOG.info("Output cnt: %d", output_cnt);
        auto [grid_h, grid_w] = input_data->get_grid()[0];
        cv::Mat mask(grid_h, grid_w, CV_8UC1); 
        assert(output_cnt == grid_h * grid_w * OBJ_CLASS_NUM);

        for (int i = 0; i < grid_h; ++i) {
            for (int j = 0; j < grid_w; ++j) {
                int max_cls = 0;
                float max_score = -1e10;
                for (int k = 0; k < OBJ_CLASS_NUM; ++k) {
                    int base_idx = (i * grid_w + j) * OBJ_CLASS_NUM;
                    float score = output_data[base_idx + k];
                    if (score > max_score) {
                        max_score = score;
                        max_cls = k;
                    }
                }
                mask.at<uchar>(i, j) = static_cast<uchar>(max_cls);;
            }
        }
        // 上采样到模型输入尺寸
        cv::Mat upsampled_mask;
        try {
            cv::resize(mask, upsampled_mask, cv::Size(model_in_w, model_in_h), 0, 0, cv::INTER_NEAREST);
        } catch (const cv::Exception& e) {
            LOG.error("OpenCV exception during resize: %s", e.what());
            return nullptr;
        }

        return std::make_shared<MaskPackage>(img_id, upsampled_mask);
    }

}
