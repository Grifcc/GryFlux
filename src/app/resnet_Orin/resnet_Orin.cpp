#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/opencv.hpp>

#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "consumer/resnet_result_consumer.h"
#include "context/orin_context.h"
#include "context/trt_model_handle.h"
#include "nodes/resnet_nodes.h"
#include "source/resnet_data_source.h"

namespace fs = std::filesystem;

namespace {

using GroundTruthMap = std::map<std::string, int>;

constexpr size_t kThreadPoolSize = 8;
constexpr size_t kMaxActivePackets = 16;
constexpr size_t kOrinContextInstances = 2;
constexpr int kOrinDeviceId = 0;
constexpr const char* kOrinResourceType = "tensorrt_gpu";

struct AppConfig {
    std::string engine_path;
    std::string dataset_dir;
    std::string gt_file_path;
};

struct AppRuntime {
    std::shared_ptr<ResNetDataSource> source;
    std::shared_ptr<ResNetResultConsumer> consumer;
    std::shared_ptr<TrtModelHandle> model_handle;
    std::shared_ptr<GryFlux::ResourcePool> resource_pool;
    std::shared_ptr<GryFlux::GraphTemplate> graph_template;
};

void InitializeLogger() {
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("resnet_Orin");
}

void PrintUsage(const char* program_name) {
    std::cout << "用法: " << program_name
              << " <engine_path> <dataset_dir> <gt_file_path>" << std::endl;
}

bool ParseArgs(int argc, char* argv[], AppConfig* config) {
    if (argc != 4 || config == nullptr) {
        return false;
    }

    config->engine_path = argv[1];
    config->dataset_dir = argv[2];
    config->gt_file_path = argv[3];
    return true;
}

bool IsImageFile(const fs::path& path);

std::unordered_set<std::string> CollectDatasetImages(const fs::path& dataset_path) {
    std::unordered_set<std::string> dataset_images;
    for (const auto& entry : fs::recursive_directory_iterator(dataset_path)) {
        if (!entry.is_regular_file() || !IsImageFile(entry.path())) {
            continue;
        }

        dataset_images.insert(fs::relative(entry.path(), dataset_path).generic_string());
    }
    return dataset_images;
}

bool IsImageFile(const fs::path& path) {
    if (!path.has_extension()) {
        return false;
    }

    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return ext == ".jpeg" || ext == ".jpg" || ext == ".png";
}

void PrintSampleEntries(const std::vector<std::string>& entries, const std::string& title) {
    if (entries.empty()) {
        return;
    }

    constexpr size_t kMaxSamplesToPrint = 10;
    std::cerr << title << std::endl;
    for (size_t i = 0; i < std::min(entries.size(), kMaxSamplesToPrint); ++i) {
        std::cerr << "  - " << entries[i] << std::endl;
    }
    if (entries.size() > kMaxSamplesToPrint) {
        std::cerr << "  ... 还有 " << (entries.size() - kMaxSamplesToPrint) << " 项未展示" << std::endl;
    }
}

GroundTruthMap LoadGroundTruth(const std::string& gt_file_path) {
    GroundTruthMap gt_map;
    std::ifstream ifs(gt_file_path);
    if (!ifs.is_open()) {
        std::cerr << "无法打开 GT 文件: " << gt_file_path << std::endl;
        return gt_map;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        const size_t split_pos = line.find(' ');
        if (split_pos == std::string::npos) {
            continue;
        }

        const std::string filename = line.substr(0, split_pos);
        const int label_index = std::stoi(line.substr(split_pos + 1));
        gt_map[filename] = label_index;
    }
    return gt_map;
}

bool ValidateGroundTruthAgainstDataset(const std::string& dataset_dir,
                                       const GroundTruthMap& gt_map) {
    const fs::path dataset_path(dataset_dir);
    if (!fs::exists(dataset_path) || !fs::is_directory(dataset_path)) {
        std::cerr << "数据集目录不存在或不是目录: " << dataset_dir << std::endl;
        return false;
    }

    const auto dataset_images = CollectDatasetImages(dataset_path);

    std::vector<std::string> missing_on_disk;
    missing_on_disk.reserve(gt_map.size());
    for (const auto& [relative_path, _] : gt_map) {
        if (dataset_images.find(relative_path) == dataset_images.end()) {
            missing_on_disk.push_back(relative_path);
        }
    }

    std::vector<std::string> missing_in_gt;
    missing_in_gt.reserve(dataset_images.size());
    for (const auto& relative_path : dataset_images) {
        if (gt_map.find(relative_path) == gt_map.end()) {
            missing_in_gt.push_back(relative_path);
        }
    }

    if (missing_on_disk.empty() && missing_in_gt.empty()) {
        std::cout << "GT 校验通过, 图片数: " << dataset_images.size() << std::endl;
        return true;
    }

    std::sort(missing_on_disk.begin(), missing_on_disk.end());
    std::sort(missing_in_gt.begin(), missing_in_gt.end());

    std::cerr << "GT 文件与数据集目录不匹配" << std::endl;
    std::cerr << "  数据集目录图片数: " << dataset_images.size() << std::endl;
    std::cerr << "  GT 条目数: " << gt_map.size() << std::endl;
    std::cerr << "  GT 中存在但目录中不存在的图片数: " << missing_on_disk.size() << std::endl;
    std::cerr << "  目录中存在但 GT 中未标注的图片数: " << missing_in_gt.size() << std::endl;

    PrintSampleEntries(missing_on_disk, "GT 中存在但目录中不存在的示例:");
    PrintSampleEntries(missing_in_gt, "目录中存在但 GT 中未标注的示例:");
    return false;
}

std::shared_ptr<TrtModelHandle> CreateModelHandle(const AppConfig& config) {
    return std::make_shared<TrtModelHandle>(kOrinDeviceId, config.engine_path);
}

std::shared_ptr<GryFlux::ResourcePool> CreateResourcePool(const std::shared_ptr<TrtModelHandle>& model_handle) {
    auto resource_pool = std::make_shared<GryFlux::ResourcePool>();

    std::vector<std::shared_ptr<GryFlux::Context>> orin_contexts;
    orin_contexts.reserve(kOrinContextInstances);
    for (size_t i = 0; i < kOrinContextInstances; ++i) {
        orin_contexts.push_back(std::make_shared<OrinContext>(kOrinDeviceId, model_handle));
    }

    resource_pool->registerResourceType(kOrinResourceType, std::move(orin_contexts));
    return resource_pool;
}

std::shared_ptr<GryFlux::GraphTemplate> BuildGraphTemplate() {
    auto graph_template = GryFlux::GraphTemplate::buildOnce(
        [](GryFlux::TemplateBuilder* builder) {
            builder->setInputNode<PreprocessNode>("preprocess");
            builder->addTask<ResNetInferNode>("inference", kOrinResourceType, {"preprocess"});
            builder->setOutputNode<PostprocessNode>("postprocess", {"inference"});
        });
    return graph_template;
}

std::shared_ptr<ResNetDataSource> CreateSource(const AppConfig& config, const GroundTruthMap& gt_map) {
    return std::make_shared<ResNetDataSource>(config.dataset_dir, gt_map);
}

std::shared_ptr<ResNetResultConsumer> CreateConsumer(size_t total_images) {
    return std::make_shared<ResNetResultConsumer>(total_images);
}

AppRuntime CreateRuntime(const AppConfig& config, const GroundTruthMap& gt_map) {
    AppRuntime runtime;
    runtime.source = CreateSource(config, gt_map);
    runtime.consumer = CreateConsumer(gt_map.size());
    runtime.model_handle = CreateModelHandle(config);
    runtime.resource_pool = CreateResourcePool(runtime.model_handle);
    runtime.graph_template = BuildGraphTemplate();
    return runtime;
}

int RunApp(const AppConfig& config) {
    cv::setNumThreads(0);
    const auto gt_map = LoadGroundTruth(config.gt_file_path);
    if (gt_map.empty()) {
        return 1;
    }
    if (!ValidateGroundTruthAgainstDataset(config.dataset_dir, gt_map)) {
        return 1;
    }

    auto runtime = CreateRuntime(config, gt_map);

    LOG.info("开始异步评估");

    GryFlux::AsyncPipeline pipeline(
        runtime.source,
        runtime.graph_template,
        runtime.resource_pool,
        runtime.consumer,
        kThreadPoolSize,
        kMaxActivePackets);

    pipeline.run();

    runtime.consumer->printMetrics();
    LOG.info("异步评估结束");
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    InitializeLogger();

    AppConfig config;
    if (!ParseArgs(argc, argv, &config)) {
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        return RunApp(config);
    } catch (const std::exception& ex) {
        LOG.error("resnet_Orin failed: %s", ex.what());
        return 1;
    }
}
