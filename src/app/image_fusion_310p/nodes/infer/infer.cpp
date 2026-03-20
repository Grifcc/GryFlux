#include "infer.h"
#include "packet/fusion_data_packet.h"
#include "context/infercontext.h"
#include <iostream>

void InferNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &fusion_packet = static_cast<FusionDataPacket&>(packet);
    auto &infer_ctx = static_cast<InferContext&>(ctx);
    infer_ctx.bindCurrentThread();  // 确保当前线程绑定到正确的 Device 上

    size_t inputSize = infer_ctx.GetInputSize();
    
    aclError ret = aclrtMemcpy(infer_ctx.GetVisDevPtr(), inputSize, 
                               fusion_packet.vis_y_float.data, inputSize, 
                               ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[InferNode] H2D Memcpy failed for Vis: " << ret << std::endl;
        return;
    }

    ret = aclrtMemcpy(infer_ctx.GetIrDevPtr(), inputSize, 
                      fusion_packet.ir_float.data, inputSize, 
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[InferNode] H2D Memcpy failed for Ir: " << ret << std::endl;
        return;
    }

    ret = aclmdlExecute(infer_ctx.GetModelId(), infer_ctx.GetInputDataset(), infer_ctx.GetOutputDataset());
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[InferNode] Model execution failed: " << ret << std::endl;
        return;
    }

    size_t outputSize = infer_ctx.GetOutputSize();
    
    ret = aclrtMemcpy(fusion_packet.fused_y_float.data, outputSize, 
                      infer_ctx.GetOutDevPtr(), outputSize, 
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "[InferNode] D2H Memcpy failed for Output: " << ret << std::endl;
        return;
    }
}