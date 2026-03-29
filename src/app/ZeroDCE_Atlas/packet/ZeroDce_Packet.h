#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <future>
#include "acl/acl.h"
#include "framework/data_packet.h"
#include <string>

// 如果 GryFlux 有基础的 Packet 类，请继承它，例如：: public gryflux::BasePacket
struct ZeroDcePacket : public GryFlux::DataPacket {
    uint64_t frame_id;
    
    // 框架强制要求的纯虚函数实现
    uint64_t getIdx() const override { return frame_id; }
    
    // 图像载体
    cv::Mat input_image;
    cv::Mat output_image;
    
    // 模型固定尺寸 (640x480, 1x3x480x640)
    size_t data_size = 1 * 3 * 480 * 640 * sizeof(float);

    // 锁页内存 (Host) 和 设备内存 (Device)
    void* host_input_ptr = nullptr;  
    void* host_output_ptr = nullptr;
    void* dev_input_ptr = nullptr;
    void* dev_output_ptr = nullptr;

    // 异步控制
    std::shared_ptr<std::promise<void>> completion_promise;

    // 构造时分配 ACL 内存 (内存池化)
    ZeroDcePacket() {
        aclrtMallocHost(&host_input_ptr, data_size);
        aclrtMallocHost(&host_output_ptr, data_size);
        aclrtMalloc(&dev_input_ptr, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dev_output_ptr, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 析构时自动释放，防止内存泄漏
    ~ZeroDcePacket() {
        if(host_input_ptr) aclrtFreeHost(host_input_ptr);
        if(host_output_ptr) aclrtFreeHost(host_output_ptr);
        if(dev_input_ptr) aclrtFree(dev_input_ptr);
        if(dev_output_ptr) aclrtFree(dev_output_ptr);
    }

    std::string image_name; 
    double int8_psnr = 0.0;
    double loss = 0.0;
    std::string status = "✅"; // 默认成功
    
};
using ZeroDcePacketPtr = std::shared_ptr<ZeroDcePacket>;