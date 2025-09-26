#pragma once

#include <cstdint>
#include <queue>
#include <vector>

#include "rknn_api.h"

class rknn_fp
{
public:
    rknn_fp(const char *model_path, int cpu_id, rknn_core_mask core_mask, int n_input, int n_output);
    ~rknn_fp();

    void dump_tensor_attr(rknn_tensor_attr *attr);
    int inference(unsigned char *data);
    float cal_NPU_performance(std::queue<float> &history_time, float &sum_time, float cost_time);

    int _cpu_id;
    int _n_input;
    int _n_output;
    rknn_context ctx;
    std::vector<rknn_tensor_attr> _input_attrs;
    std::vector<rknn_tensor_attr> _output_attrs;
    std::vector<rknn_tensor_mem *> _input_mems;
    std::vector<rknn_tensor_mem *> _output_mems;
    std::vector<float *> _output_buff;
};