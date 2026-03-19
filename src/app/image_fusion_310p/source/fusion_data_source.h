#pragma once
#include "framework/data_source.h"
#include <string>
#include <filesystem>

class FusionDataSource : public GryFlux::DataSource {
public:
    FusionDataSource(const std::string& visDir, const std::string& irDir);
    ~FusionDataSource() override = default;
    std::unique_ptr<GryFlux::DataPacket> produce() override;

private:
    std::string irDir_;
    std::filesystem::directory_iterator dirIter_;
    std::filesystem::directory_iterator endIter_;
    uint64_t current_idx_ = 0; // 添加计数器
};