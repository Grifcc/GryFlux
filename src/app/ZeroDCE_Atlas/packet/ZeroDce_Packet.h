#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <future>
#include "acl/acl.h"
#include "framework/data_packet.h"
#include <string>

struct ZeroDcePacket : public GryFlux::DataPacket {
    uint64_t frame_id;
    
    uint64_t getIdx() const override { return frame_id; }
    
    cv::Mat input_image;
    cv::Mat output_image;
    
    size_t data_size = 1 * 3 * 480 * 640 * sizeof(float);

    void* host_input_ptr = nullptr;  
    void* host_output_ptr = nullptr;
    void* dev_input_ptr = nullptr;
    void* dev_output_ptr = nullptr;

    std::shared_ptr<std::promise<void>> completion_promise;

    ZeroDcePacket() {
        aclrtMallocHost(&host_input_ptr, data_size);
        aclrtMallocHost(&host_output_ptr, data_size);
        aclrtMalloc(&dev_input_ptr, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dev_output_ptr, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    ~ZeroDcePacket() {
        if(host_input_ptr) aclrtFreeHost(host_input_ptr);
        if(host_output_ptr) aclrtFreeHost(host_output_ptr);
        if(dev_input_ptr) aclrtFree(dev_input_ptr);
        if(dev_output_ptr) aclrtFree(dev_output_ptr);
    }

    std::string image_name; 
    double int8_psnr = 0.0;
    double loss = 0.0;
    std::string status = "✅"; 
    
};
using ZeroDcePacketPtr = std::shared_ptr<ZeroDcePacket>;