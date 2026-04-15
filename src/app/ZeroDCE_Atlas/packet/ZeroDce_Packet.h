#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "framework/data_packet.h"
#include <string>

struct ZeroDcePacket : public GryFlux::DataPacket {
    uint64_t frame_id;
    
    uint64_t getIdx() const override { return frame_id; }
    
    cv::Mat input_image;
    cv::Mat output_image;
    std::vector<float> input_tensor;
    std::vector<unsigned char> host_output_buffer;

    void* dev_input_ptr = nullptr;
    void* dev_output_ptr = nullptr;

    size_t input_size = 0;
    size_t output_size = 0;

    void EnsureBuffers(size_t new_input_size, size_t new_output_size) {
        if (input_size == new_input_size && output_size == new_output_size &&
            dev_input_ptr != nullptr && dev_output_ptr != nullptr) {
            return;
        }

        if (dev_input_ptr) {
            aclrtFree(dev_input_ptr);
            dev_input_ptr = nullptr;
        }
        if (dev_output_ptr) {
            aclrtFree(dev_output_ptr);
            dev_output_ptr = nullptr;
        }

        input_size = new_input_size;
        output_size = new_output_size;
        host_output_buffer.resize(output_size);

        aclError ret = aclrtMalloc(&dev_input_ptr, input_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            input_size = 0;
            output_size = 0;
            throw std::runtime_error("aclrtMalloc failed for dev_input_ptr");
        }

        ret = aclrtMalloc(&dev_output_ptr, output_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclrtFree(dev_input_ptr);
            dev_input_ptr = nullptr;
            input_size = 0;
            output_size = 0;
            throw std::runtime_error("aclrtMalloc failed for dev_output_ptr");
        }
    }

    ~ZeroDcePacket() {
        if(dev_input_ptr) aclrtFree(dev_input_ptr);
        if(dev_output_ptr) aclrtFree(dev_output_ptr);
    }

    std::string image_name; 
    double int8_psnr = 0.0;
    double loss = 0.0;
    std::string status = "OK";
    
};
using ZeroDcePacketPtr = std::shared_ptr<ZeroDcePacket>;
