#pragma once

#include <string_view>
#include "framework/processing_task.h"
#include "fstream"
#include "utils/logger.h"
#include "rknn_api.h"

namespace GryFlux
{
    using ModelData = std::pair<std::unique_ptr<unsigned char[]>, std::size_t>; //buffer, size

    class RkRunner: public ProcessingTask
    {
    public:
    	explicit RkRunner(std::string_view model_path, const int npu_id = 1,
                          const std::size_t model_width = 640, const std::size_t model_height = 640);
        std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;

        // Function to read binary file into a buffer allocated in memory
        std::optional<ModelData> load_model(std::string_view filename);

        void dump_tensor_attr(rknn_tensor_attr* attr);
        inline float deqnt_affine_to_f32(int8_t qnt, int zp, float scale);
        ~RkRunner();


    private:
        rknn_context rknn_ctx_;
        std::size_t input_num_;
        std::size_t output_num_;
        rknn_tensor_attr* input_attrs_;
        rknn_tensor_attr* output_attrs_;
        //zero copy buffer for rknn 
        std::vector<rknn_tensor_mem*> input_mems_;
        std::vector<rknn_tensor_mem*> output_mems_;
        // rknn_tensor_mem** input_mems_;
        // rknn_tensor_mem** output_mems_;
        std::size_t model_width_;
        std::size_t model_height_;
        bool is_quant_;
    };
}
