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
    explicit RealEsrganSource(const std::string &inputDir)
    {
        std::cout << "[Source] Scanning input directory: " << inputDir << std::endl;
        for (const auto &entry : std::filesystem::directory_iterator(inputDir))
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

        ++currentIdx_;
        setHasMore(currentIdx_ < lrFiles_.size());
        return packet;
    }

    int getTotalImages() const
    {
        return static_cast<int>(lrFiles_.size());
    }

private:
    std::vector<std::filesystem::path> lrFiles_;
    size_t currentIdx_ = 0;
};
