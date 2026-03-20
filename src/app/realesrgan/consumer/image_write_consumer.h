#pragma once

#include "framework/data_consumer.h"
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

class ImageWriteConsumer : public GryFlux::DataConsumer
{
public:
    explicit ImageWriteConsumer(const std::string &outputDir);

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

    size_t getWrittenCount() const { return writtenCount_; }

private:
    fs::path outputDir_;
    size_t writtenCount_ = 0;
};
