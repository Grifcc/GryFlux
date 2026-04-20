#include "PostprocessNode.h"

#include "../../consumer/DiskWriter/AsyncDiskWriter.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>

namespace {

double ComputeMeanLuma(const cv::Mat& bgr_image) {
    cv::Mat gray;
    cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);
    return cv::mean(gray)[0];
}

inline float ClampToUnit(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

double ComputeScaledLoss(const cv::Mat& lhs_bgr, const cv::Mat& rhs_bgr) {
    cv::Mat lhs_float;
    cv::Mat rhs_float;
    lhs_bgr.convertTo(lhs_float, CV_32FC3, 1.0 / 255.0);
    rhs_bgr.convertTo(rhs_float, CV_32FC3, 1.0 / 255.0);

    cv::Mat diff = lhs_float - rhs_float;
    cv::Mat squared;
    cv::multiply(diff, diff, squared);
    const cv::Scalar mse_scalar = cv::mean(squared);
    const double mse = (mse_scalar[0] + mse_scalar[1] + mse_scalar[2]) / 3.0;
    return mse * 100.0;
}

double ComputePsnr(const cv::Mat& lhs_bgr, const cv::Mat& rhs_bgr) {
    cv::Mat lhs_float;
    cv::Mat rhs_float;
    lhs_bgr.convertTo(lhs_float, CV_32FC3, 1.0 / 255.0);
    rhs_bgr.convertTo(rhs_float, CV_32FC3, 1.0 / 255.0);

    cv::Mat diff = lhs_float - rhs_float;
    cv::Mat squared;
    cv::multiply(diff, diff, squared);
    const cv::Scalar mse_scalar = cv::mean(squared);
    const double mse = (mse_scalar[0] + mse_scalar[1] + mse_scalar[2]) / 3.0;
    if (mse <= 1e-12) {
        return 99.99;
    }
    return 10.0 * std::log10(1.0 / mse);
}

}  // namespace

void PostprocessNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& dce_packet = static_cast<ZeroDcePacket&>(packet);
    (void)ctx;
    const auto start_time = std::chrono::steady_clock::now();

    if (!dce_packet.is_valid_image) {
        dce_packet.write_enqueued = false;
        const auto end_time = std::chrono::steady_clock::now();
        dce_packet.postprocess_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return;
    }

    if (dce_packet.infer_only) {
        dce_packet.output_image.release();
        dce_packet.write_enqueued = false;
        dce_packet.output_mean_luma = 0.0;
        dce_packet.loss = 0.0;
        dce_packet.int8_psnr = 0.0;
        dce_packet.is_proxy_psnr = true;
        dce_packet.status = "INFER";
        const auto end_time = std::chrono::steady_clock::now();
        dce_packet.postprocess_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return;
    }

    const int image_area = dce_packet.output_width * dce_packet.output_height;
    const float* channel_r = dce_packet.output_tensor.data();
    const float* channel_g = channel_r + image_area;
    const float* channel_b = channel_g + image_area;
    std::vector<cv::Mat> bgr_channels;
    bgr_channels.reserve(3);
    bgr_channels.emplace_back(
        dce_packet.output_height, dce_packet.output_width, CV_32FC1, const_cast<float*>(channel_b));
    bgr_channels.emplace_back(
        dce_packet.output_height, dce_packet.output_width, CV_32FC1, const_cast<float*>(channel_g));
    bgr_channels.emplace_back(
        dce_packet.output_height, dce_packet.output_width, CV_32FC1, const_cast<float*>(channel_r));

    cv::Mat enhanced_bgr_float;
    cv::merge(bgr_channels, enhanced_bgr_float);
    cv::min(enhanced_bgr_float, 1.0, enhanced_bgr_float);
    cv::max(enhanced_bgr_float, 0.0, enhanced_bgr_float);

    cv::Mat enhanced_bgr;
    enhanced_bgr_float.convertTo(enhanced_bgr, CV_8UC3, 255.0);

    if (!dce_packet.input_image.empty() &&
        (dce_packet.input_image.cols != enhanced_bgr.cols ||
         dce_packet.input_image.rows != enhanced_bgr.rows)) {
        cv::resize(
            enhanced_bgr,
            dce_packet.output_image,
            dce_packet.input_image.size(),
            0.0,
            0.0,
            cv::INTER_LINEAR);
    } else {
        dce_packet.output_image = enhanced_bgr;
    }

    if (dce_packet.enable_metrics) {
        dce_packet.output_mean_luma = ComputeMeanLuma(dce_packet.output_image);
        dce_packet.is_proxy_psnr = true;
        cv::Mat aligned_input;
        cv::resize(
            dce_packet.input_image,
            aligned_input,
            dce_packet.output_image.size(),
            0.0,
            0.0,
            cv::INTER_LINEAR);

        if (dce_packet.has_ground_truth && !dce_packet.gt_path.empty()) {
            cv::Mat gt_image = cv::imread(dce_packet.gt_path, cv::IMREAD_COLOR);
            if (!gt_image.empty()) {
                cv::Mat aligned_gt;
                cv::resize(
                    gt_image,
                    aligned_gt,
                    dce_packet.output_image.size(),
                    0.0,
                    0.0,
                    cv::INTER_LINEAR);
                dce_packet.loss = ComputeScaledLoss(aligned_gt, dce_packet.output_image);
                dce_packet.int8_psnr = ComputePsnr(aligned_gt, dce_packet.output_image);
                dce_packet.is_proxy_psnr = false;
            } else {
                dce_packet.loss = ComputeScaledLoss(aligned_input, dce_packet.output_image);
                dce_packet.int8_psnr = ComputePsnr(aligned_input, dce_packet.output_image);
            }
        } else {
            dce_packet.loss = ComputeScaledLoss(aligned_input, dce_packet.output_image);
            dce_packet.int8_psnr = ComputePsnr(aligned_input, dce_packet.output_image);
        }
    } else {
        dce_packet.output_mean_luma = 0.0;
        dce_packet.loss = 0.0;
        dce_packet.int8_psnr = 0.0;
        dce_packet.is_proxy_psnr = true;
    }

    if (dce_packet.enable_save) {
        AsyncDiskWriter::GetInstance().Push(dce_packet.image_name, dce_packet.output_image);
    }

    dce_packet.write_enqueued = dce_packet.enable_save;
    if (!dce_packet.enable_metrics) {
        dce_packet.status = dce_packet.enable_save ? "OK" : "BENCH";
    } else {
        dce_packet.status = dce_packet.loss > 4.5 ? "WARN" : "OK";
    }

    const auto end_time = std::chrono::steady_clock::now();
    dce_packet.postprocess_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
}
