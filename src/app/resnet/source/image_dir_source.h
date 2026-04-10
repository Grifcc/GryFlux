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

class ResnetImageDirSource : public GryFlux::DataSource
{
public:
    explicit ResnetImageDirSource(const std::string &datasetDir);

    std::unique_ptr<GryFlux::DataPacket> produce() override;

private:
    static bool isSupportedImage(const fs::path &path);

    fs::path datasetDir_;
    std::vector<fs::path> imageFiles_;
    std::size_t cursor_ = 0;
    int idx_ = 0;
};

