#pragma once

#include "framework/data_source.h"
#include "packet/realesrgan_packet.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

class RealEsrganSource : public GryFlux::DataSource
{
public:
    RealEsrganSource(const std::string &lrDir, const std::string &hrDir)
        : hrDir_(hrDir)
    {
        std::cout << "[Source] Scanning low-resolution directory: " << lrDir << std::endl;
        for (const auto &entry : std::filesystem::directory_iterator(lrDir))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            const auto ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
            {
                lrFiles_.push_back(entry.path());
            }
        }
        std::sort(lrFiles_.begin(), lrFiles_.end());
        std::cout << "[Source] Found " << lrFiles_.size() << " low-resolution images." << std::endl;
        setHasMore(!lrFiles_.empty());
    }

    std::unique_ptr<GryFlux::DataPacket> produce() override
    {
        if (currentIdx_ >= lrFiles_.size())
        {
            setHasMore(false);
            return nullptr;
        }

        auto packet = std::make_unique<RealEsrganPacket>();
        packet->frame_id = static_cast<int>(currentIdx_);
        packet->lr_path = lrFiles_[currentIdx_].string();
        packet->hr_path = buildHrPath(lrFiles_[currentIdx_]).string();

        ++currentIdx_;
        setHasMore(currentIdx_ < lrFiles_.size());
        return packet;
    }

    int getTotalImages() const
    {
        return static_cast<int>(lrFiles_.size());
    }

private:
    std::filesystem::path buildHrPath(const std::filesystem::path &lrPath) const
    {
        std::string fileName = lrPath.filename().string();
        const auto pos = fileName.find("_LR");
        if (pos != std::string::npos)
        {
            fileName.replace(pos, 3, "_HR");
        }
        return std::filesystem::path(hrDir_) / fileName;
    }

    std::vector<std::filesystem::path> lrFiles_;
    std::string hrDir_;
    size_t currentIdx_ = 0;
};
