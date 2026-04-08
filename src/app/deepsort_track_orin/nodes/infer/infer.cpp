#include "infer.h"
#include "../../packet/track_data_packet.h"
#include "../../context/infercontext.h"
#include <iostream>
#include <cstring> 

void InferNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &p = static_cast<TrackDataPacket&>(packet);
    auto &infer_ctx = static_cast<InferContext&>(ctx);

    infer_ctx.bindCurrentThread();

    size_t data_bytes = p.preproc_data.nchw_data.size() * sizeof(float);
    if (data_bytes != infer_ctx.getInputBufferSize()) {
        std::cerr << "[InferNode] 错误: 帧 " << p.frame_id 
                  << " 预处理数据大小 (" << data_bytes 
                  << " 字节) 与模型期望的输入大小 (" << infer_ctx.getInputBufferSize() 
                  << " 字节) 不匹配！" << std::endl;
        return;
    }

    infer_ctx.copyToDevice(p.preproc_data.nchw_data.data(), data_bytes);

    infer_ctx.executeModel();

    infer_ctx.copyToHost();

    size_t num_outputs = infer_ctx.getNumOutputs();
    if (p.infer_outputs.size() < num_outputs) {
        p.infer_outputs.resize(num_outputs);
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        float* host_ptr = static_cast<float*>(infer_ctx.getOutputHostBuffer(i));
        size_t float_count = infer_ctx.getOutputSize(i) / sizeof(float);
        
        p.infer_outputs[i].assign(host_ptr, host_ptr + float_count);
    }
}
