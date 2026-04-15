#include "InferNode.h"
#include <cstring>

namespace {

void CheckAcl(aclError ret, const char* expr) {
    if (ret == ACL_SUCCESS) return;
    throw std::runtime_error(std::string("ACL call failed: ") + expr);
}

}

void InferNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto& dce_packet = dynamic_cast<ZeroDcePacket&>(packet);
    auto& atlas_ctx = dynamic_cast<AtlasContext&>(ctx);

    aclmdlDesc* model_desc = atlas_ctx.GetModelDesc();
    if (model_desc == nullptr) {
        throw std::runtime_error("model desc is null");
    }

    size_t input_size = aclmdlGetInputSizeByIndex(model_desc, 0);
    size_t output_size = aclmdlGetOutputSizeByIndex(model_desc, 0);

    if (dce_packet.input_tensor.size() * sizeof(float) != input_size) {
        throw std::runtime_error("input tensor size mismatch");
    }

    CheckAcl(aclrtSetCurrentContext(atlas_ctx.GetAclContext()), "aclrtSetCurrentContext");
    dce_packet.EnsureBuffers(input_size, output_size);

    CheckAcl(aclrtMemcpy(dce_packet.dev_input_ptr, input_size,
                         dce_packet.input_tensor.data(), input_size,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(H2D)");

    aclmdlDataset *inputDataset = nullptr;
    aclDataBuffer *inputBuffer = nullptr;
    aclmdlDataset *outputDataset = nullptr;
    aclDataBuffer *outputBuffer = nullptr;

    try {
        inputDataset = aclmdlCreateDataset();
        if (inputDataset == nullptr) {
            throw std::runtime_error("aclmdlCreateDataset failed for input");
        }
        inputBuffer = aclCreateDataBuffer(dce_packet.dev_input_ptr, input_size);
        if (inputBuffer == nullptr) {
            throw std::runtime_error("aclCreateDataBuffer failed for input");
        }
        CheckAcl(aclmdlAddDatasetBuffer(inputDataset, inputBuffer), "aclmdlAddDatasetBuffer(input)");

        outputDataset = aclmdlCreateDataset();
        if (outputDataset == nullptr) {
            throw std::runtime_error("aclmdlCreateDataset failed for output");
        }
        outputBuffer = aclCreateDataBuffer(dce_packet.dev_output_ptr, output_size);
        if (outputBuffer == nullptr) {
            throw std::runtime_error("aclCreateDataBuffer failed for output");
        }
        CheckAcl(aclmdlAddDatasetBuffer(outputDataset, outputBuffer), "aclmdlAddDatasetBuffer(output)");

        CheckAcl(aclmdlExecute(atlas_ctx.GetModelId(), inputDataset, outputDataset), "aclmdlExecute");
    } catch (...) {
        if (inputBuffer != nullptr) aclDestroyDataBuffer(inputBuffer);
        if (inputDataset != nullptr) aclmdlDestroyDataset(inputDataset);
        if (outputBuffer != nullptr) aclDestroyDataBuffer(outputBuffer);
        if (outputDataset != nullptr) aclmdlDestroyDataset(outputDataset);
        throw;
    }

    aclDestroyDataBuffer(inputBuffer);
    aclmdlDestroyDataset(inputDataset);
    aclDestroyDataBuffer(outputBuffer);
    aclmdlDestroyDataset(outputDataset);

    CheckAcl(aclrtMemcpy(dce_packet.host_output_buffer.data(), output_size,
                         dce_packet.dev_output_ptr, output_size,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(D2H)");
}
