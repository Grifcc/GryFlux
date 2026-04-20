#include "ZeroDceDataSource.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

bool IsImageFile(const fs::path& path) {
    if (!path.has_extension()) {
        return false;
    }

    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

void LogFilesystemWarning(const fs::path& root,
                          const std::error_code& ec) {
    if (ec) {
        std::cerr << "[Source] 跳过无法访问的路径 (" << root.string()
                  << "): " << ec.message() << std::endl;
    }
}

}  // namespace

ZeroDceDataSource::ZeroDceDataSource(const std::string& input_dir,
                                     const std::string& gt_dir,
                                     bool enable_save,
                                     bool enable_metrics,
                                     bool infer_only,
                                     int input_channels,
                                     int input_height,
                                     int input_width,
                                     int output_channels,
                                     int output_height,
                                     int output_width)
    : input_channels_(input_channels),
      input_height_(input_height),
      input_width_(input_width),
      output_channels_(output_channels),
      output_height_(output_height),
      output_width_(output_width),
      input_root_(input_dir),
      gt_root_(gt_dir),
      enable_save_(enable_save),
      enable_metrics_(enable_metrics),
      infer_only_(infer_only) {
    if (!fs::exists(input_root_) || !fs::is_directory(input_root_)) {
        throw std::runtime_error("ZeroDCE input_dir does not exist or is not a directory: " + input_dir);
    }

    std::error_code input_iter_ec;
    for (fs::recursive_directory_iterator it(
             input_root_, fs::directory_options::skip_permission_denied, input_iter_ec),
         end;
         it != end;
         it.increment(input_iter_ec)) {
        LogFilesystemWarning(input_root_, input_iter_ec);
        input_iter_ec.clear();

        std::error_code status_ec;
        if (!it->is_regular_file(status_ec)) {
            LogFilesystemWarning(it->path(), status_ec);
            continue;
        }
        if (!IsImageFile(it->path())) {
            continue;
        }
        image_paths_.push_back(it->path().string());
    }

    if (!gt_dir.empty()) {
        if (!fs::exists(gt_root_) || !fs::is_directory(gt_root_)) {
            throw std::runtime_error("ZeroDCE gt_dir does not exist or is not a directory: " + gt_dir);
        }

        std::error_code gt_iter_ec;
        for (fs::recursive_directory_iterator it(
                 gt_root_, fs::directory_options::skip_permission_denied, gt_iter_ec),
             end;
             it != end;
             it.increment(gt_iter_ec)) {
            LogFilesystemWarning(gt_root_, gt_iter_ec);
            gt_iter_ec.clear();

            std::error_code status_ec;
            if (!it->is_regular_file(status_ec)) {
                LogFilesystemWarning(it->path(), status_ec);
                continue;
            }
            if (!IsImageFile(it->path())) {
                continue;
            }

            const std::string relative_path = fs::relative(it->path(), gt_root_).generic_string();
            gt_by_relative_path_[relative_path] = it->path().string();
            gt_by_filename_[it->path().filename().string()] = it->path().string();
        }
    }

    std::sort(image_paths_.begin(), image_paths_.end());
    std::cout << "[Source] 发现待处理图片数量: " << image_paths_.size() << std::endl;
}

std::unique_ptr<GryFlux::DataPacket> ZeroDceDataSource::produce() {
    if (current_idx_ >= image_paths_.size()) {
        setHasMore(false);
        return nullptr;
    }

    auto packet = std::make_unique<ZeroDcePacket>(
        input_channels_,
        input_height_,
        input_width_,
        output_channels_,
        output_height_,
        output_width_);
    packet->frame_id = current_idx_;
    packet->image_path = image_paths_[current_idx_];
    packet->image_name = "image_" + std::to_string(current_idx_) + ".jpg";
    packet->source_filename = fs::path(packet->image_path).filename().string();
    packet->enable_save = enable_save_;
    packet->enable_metrics = enable_metrics_;
    packet->infer_only = infer_only_;
    packet->input_image = cv::imread(packet->image_path, cv::IMREAD_COLOR);

    if (!gt_root_.empty()) {
        const std::string relative_path = fs::relative(packet->image_path, input_root_).generic_string();
        auto gt_it = gt_by_relative_path_.find(relative_path);
        if (gt_it != gt_by_relative_path_.end()) {
            packet->gt_path = gt_it->second;
            packet->has_ground_truth = true;
        } else {
            auto filename_it = gt_by_filename_.find(packet->source_filename);
            if (filename_it != gt_by_filename_.end()) {
                packet->gt_path = filename_it->second;
                packet->has_ground_truth = true;
            }
        }
    }

    ++current_idx_;
    setHasMore(current_idx_ < image_paths_.size());
    return packet;
}
