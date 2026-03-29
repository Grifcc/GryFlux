#include "InferNode.h"
#include <iostream>
#include <mutex>
#include <unordered_map>

static std::unordered_map<GryFlux::Context*, std::pair<void*, void*>> s_npu_memory_pool;
static std::mutex s_pool_mutex;

void InferNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto& dce_packet = dynamic_cast<ZeroDcePacket&>(packet);
    auto& atlas_ctx = dynamic_cast<AtlasContext&>(ctx);

    // 1. 绑定 NPU
    aclrtSetCurrentContext(atlas_ctx.GetAclContext());

    // 2. 正常申请显存
    aclrtMalloc((void**)&dce_packet.dev_input_ptr, dce_packet.data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dce_packet.dev_output_ptr, dce_packet.data_size, ACL_MEM_MALLOC_HUGE_FIRST);

    // 3. H2D 拷贝
    aclrtMemcpy(dce_packet.dev_input_ptr, dce_packet.data_size, 
                dce_packet.input_image.data, dce_packet.data_size, 
                ACL_MEMCPY_HOST_TO_DEVICE);

    // 4. 执行推理
    aclmdlDataset *inputDataset = aclmdlCreateDataset();
    aclDataBuffer *inputBuffer = aclCreateDataBuffer(dce_packet.dev_input_ptr, dce_packet.data_size);
    aclmdlAddDatasetBuffer(inputDataset, inputBuffer);

    aclmdlDataset *outputDataset = aclmdlCreateDataset();
    aclDataBuffer *outputBuffer = aclCreateDataBuffer(dce_packet.dev_output_ptr, dce_packet.data_size);
    aclmdlAddDatasetBuffer(outputDataset, outputBuffer);

    aclmdlExecute(atlas_ctx.GetModelId(), inputDataset, outputDataset);

    // 销毁轻量级结构
    aclDestroyDataBuffer(inputBuffer);
    aclmdlDestroyDataset(inputDataset);
    aclDestroyDataBuffer(outputBuffer);
    aclmdlDestroyDataset(outputDataset);

    // 5. D2H 拷贝回结果
    aclrtMemcpy(dce_packet.host_output_ptr, dce_packet.data_size, 
                dce_packet.dev_output_ptr, dce_packet.data_size, 
                ACL_MEMCPY_DEVICE_TO_HOST);

    // 6. 🌟 乖乖释放显存！这是让程序优雅退出的唯一方法 🌟
    aclrtFree(dce_packet.dev_input_ptr);
    aclrtFree(dce_packet.dev_output_ptr);
    dce_packet.dev_input_ptr = nullptr;
    dce_packet.dev_output_ptr = nullptr;
}