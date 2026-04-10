#pragma once

#include "framework/data_source.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "No filesystem support found"
#endif

#include <string>
#include <vector>

class FusionImagePairSource : public GryFlux::DataSource
{
public:
    explicit FusionImagePairSource(const std::string &datasetRoot);

    std::unique_ptr<GryFlux::DataPacket> produce() override;

private:
    static bool isSupportedImage(const fs::path &path);

    fs::path datasetRoot_;
    fs::path visibleDir_;
    fs::path infraredDir_;
    std::vector<std::string> filenames_;
    size_t cursor_ = 0;
    int idx_ = 0;
};
