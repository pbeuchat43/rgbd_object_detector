// Copyright (c) 2026 Paul N. Beuchat. MIT License. See NOTICE.md.
#include "rgbd_object_detector/core.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rgbd_object_detector {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
bool finite_between(double value, double low, double high) {
  return std::isfinite(value) && value >= low && value <= high;
}
double median(std::vector<double>& values) {
  const auto middle = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), middle, values.end());
  return values.size() % 2 ? *middle : (*middle + *std::max_element(values.begin(), middle)) / 2;
}
double iou(const Box& a, const Box& b) {
  const auto intersection = std::max(0.0, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                            std::max(0.0, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const auto total = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection;
  return total > 0 ? intersection / total : 0;
}
}

void Config::validate() const {
  require(finite_between(score_threshold, 0, 1), "score_threshold must be in [0,1]");
  require(finite_between(iou_threshold, 0, 1), "iou_threshold must be in [0,1]");
  require(pre_nms_limit >= 1 && pre_nms_limit <= kPredictions, "pre_nms_limit must be 1..8400");
  require(max_detections >= 1 && max_detections <= 100, "max_detections must be 1..100");
  require(finite_between(roi_fraction, 0.01, 1), "roi_fraction must be 0.01..1");
  require(finite_between(min_depth_m, 0.001, 100) && finite_between(max_depth_m, 0.001, 100) &&
          min_depth_m < max_depth_m, "depth bounds must be ordered in 0.001..100 metres");
  require(min_valid_pixels >= 1 && min_valid_pixels <= 1920 * 1080, "min_valid_pixels out of range");
  require(finite_between(min_valid_fraction, 0, 1), "min_valid_fraction must be in [0,1]");
  require(finite_between(max_frame_age_ms, 10, 2000), "max_frame_age_ms must be 10..2000");
}

void ImageView::validate(uint32_t channels) const {
  require(data && width > 0 && width <= 1920 && height > 0 && height <= 1080,
          "unsupported image dimensions/storage");
  require(step >= uint64_t(width) * channels && step <= uint64_t(width) * channels + 4096,
          "invalid image row stride");
  require(bytes == uint64_t(step) * height, "image byte count does not match rows/stride");
}
void DepthView::validate() const {
  image.validate(2);
  require(finite_between(unit_m, 1e-9, 1.0), "invalid depth unit");
  for (auto value : k) require(std::isfinite(value), "nonfinite intrinsics");
  require(k[0] > 0 && k[4] > 0 && k[1] == 0 && k[3] == 0 && k[6] == 0 && k[7] == 0 && k[8] == 1,
          "unsupported rectified intrinsics");
  require(k[2] >= 0 && k[2] < image.width && k[5] >= 0 && k[5] < image.height,
          "principal point outside image");
}
SpatialEstimate::SpatialEstimate()
    : xyz{NAN, NAN, NAN}, depth_mad_m(NAN) {}

void preprocess(const ImageView& bgr, float* destination, size_t elements) {
  bgr.validate(3);
  require(destination && elements == kInputElements, "incorrect input tensor storage");
  // OpenCV's input view is const in use; every operation writes separate storage.
  const cv::Mat source(bgr.height, bgr.width, CV_8UC3, const_cast<uint8_t*>(bgr.data), bgr.step);
  cv::Mat resized, rgb, normalized;
  if (bgr.width == kModelWidth && bgr.height == kModelHeight) resized = source;
  else cv::resize(source, resized, cv::Size(kModelWidth, kModelHeight), 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
  std::array<cv::Mat, 3> planes;
  for (size_t channel = 0; channel < 3; ++channel)
    planes[channel] = cv::Mat(kModelHeight, kModelWidth, CV_32F,
                             destination + channel * kModelWidth * kModelHeight);
  cv::split(normalized, planes.data());
}

Decoded decode(const float* output, size_t elements, uint32_t source_width,
               uint32_t source_height, const Config& config) {
  config.validate();
  require(output && elements == kOutputElements, "incorrect output tensor shape");
  require(source_width > 0 && source_width <= 1920 && source_height > 0 && source_height <= 1080,
          "invalid source dimensions");
  std::vector<Box> candidates;
  candidates.reserve(kPredictions);
  const double sx = double(source_width) / kModelWidth, sy = double(source_height) / kModelHeight;
  for (uint32_t index = 0; index < kPredictions; ++index) {
    std::array<double, 6> value;
    for (size_t channel = 0; channel < value.size(); ++channel) {
      value[channel] = output[channel * kPredictions + index];
      require(std::isfinite(value[channel]), "nonfinite inference output");
    }
    require(value[4] >= 0 && value[4] <= 1 && value[5] >= 0 && value[5] <= 1,
            "class score outside [0,1]");
    const uint32_t label = value[5] > value[4] ? 1 : 0;
    const auto score = value[4 + label];
    if (score < config.score_threshold || value[2] <= 0 || value[3] <= 0) continue;
    Box box{std::clamp((value[0] - value[2] / 2) * sx, 0.0, double(source_width)),
            std::clamp((value[1] - value[3] / 2) * sy, 0.0, double(source_height)),
            std::clamp((value[0] + value[2] / 2) * sx, 0.0, double(source_width)),
            std::clamp((value[1] + value[3] / 2) * sy, 0.0, double(source_height)),
            score, label, index};
    if (box.x2 > box.x1 && box.y2 > box.y1) candidates.push_back(box);
  }
  Decoded result;
  result.candidate_count = candidates.size();
  std::sort(candidates.begin(), candidates.end(), [](const Box& a, const Box& b) {
    return a.score != b.score ? a.score > b.score : a.prediction_index < b.prediction_index;
  });
  if (candidates.size() > size_t(config.pre_nms_limit)) {
    candidates.resize(config.pre_nms_limit); result.truncated = true;
  }
  result.nms_candidate_count = candidates.size();
  std::vector<Box> retained;
  retained.reserve(candidates.size());
  for (const auto& box : candidates) {
    const bool suppressed = std::any_of(retained.begin(), retained.end(), [&](const Box& other) {
      return other.class_index == box.class_index && iou(box, other) > config.iou_threshold;
    });
    if (!suppressed) retained.push_back(box);
  }
  result.nms_count = retained.size();
  if (retained.size() > size_t(config.max_detections)) {
    retained.resize(config.max_detections); result.truncated = true;
  }
  result.boxes = std::move(retained);
  return result;
}

SpatialEstimate estimate(const Box& box, const DepthView& depth, const Config& config) {
  config.validate(); depth.validate();
  require(std::isfinite(box.x1) && std::isfinite(box.x2) && std::isfinite(box.y1) && std::isfinite(box.y2),
          "nonfinite spatial box");
  require(box.x1 >= 0 && box.y1 >= 0 && box.x2 <= depth.image.width && box.y2 <= depth.image.height &&
          box.x2 > box.x1 && box.y2 > box.y1, "spatial box outside source image");
  SpatialEstimate result;
  result.u = (box.x1 + box.x2) / 2; result.v = (box.y1 + box.y2) / 2;
  const auto half_width = (box.x2 - box.x1) * config.roi_fraction / 2;
  const auto half_height = (box.y2 - box.y1) * config.roi_fraction / 2;
  const int x1 = std::max(0, int(std::ceil(result.u - half_width)));
  const int x2 = std::min(int(depth.image.width), int(std::ceil(result.u + half_width)));
  const int y1 = std::max(0, int(std::ceil(result.v - half_height)));
  const int y2 = std::min(int(depth.image.height), int(std::ceil(result.v + half_height)));
  result.roi_pixels = std::max(0, x2 - x1) * std::max(0, y2 - y1);
  if (!result.roi_pixels) { result.reason = "empty_region"; return result; }
  std::vector<double> samples;
  samples.reserve(result.roi_pixels);
  for (int y = y1; y < y2; ++y) {
    const auto* row = depth.image.data + size_t(y) * depth.image.step;
    for (int x = x1; x < x2; ++x) {
      const uint16_t raw = uint16_t(row[x * 2]) | uint16_t(row[x * 2 + 1]) << 8;
      const double z = raw * depth.unit_m;
      if (raw != depth.invalid && z >= config.min_depth_m && z <= config.max_depth_m) samples.push_back(z);
    }
  }
  result.valid_pixels = samples.size();
  result.valid_fraction = double(result.valid_pixels) / result.roi_pixels;
  if (result.valid_pixels < uint32_t(config.min_valid_pixels)) {
    result.reason = "insufficient_depth"; return result;
  }
  if (result.valid_fraction < config.min_valid_fraction) {
    result.reason = "low_valid_fraction"; return result;
  }
  const double z = median(samples);
  for (auto& sample : samples) sample = std::abs(sample - z);
  result.depth_mad_m = median(samples);
  result.xyz = {(result.u - depth.k[2]) * z / depth.k[0],
                (result.v - depth.k[5]) * z / depth.k[4], z};
  result.valid = true;
  return result;
}

int64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::string local_clock_domain() {
  std::ifstream file("/proc/sys/kernel/random/boot_id");
  std::string boot;
  if (!std::getline(file, boot) || boot.size() != 36) throw std::runtime_error("cannot identify Linux monotonic boot clock");
  return "linux_monotonic:" + boot;
}
std::string sha256_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open artifact: " + path);
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA-256 initialization failed");
  std::array<char, 65536> buffer;
  while (file) {
    file.read(buffer.data(), buffer.size());
    if (EVP_DigestUpdate(context.get(), buffer.data(), file.gcount()) != 1)
      throw std::runtime_error("SHA-256 update failed");
  }
  if (!file.eof()) throw std::runtime_error("artifact read failed: " + path);
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest;
  unsigned int count = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &count) != 1) throw std::runtime_error("SHA-256 finalization failed");
  std::ostringstream result;
  for (unsigned int i = 0; i < count; ++i) result << std::hex << std::setw(2) << std::setfill('0') << int(digest[i]);
  return result.str();
}
std::string sha256_bytes(const void* data, size_t bytes) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest;
  unsigned int count = 0;
  if (EVP_Digest(data, bytes, digest.data(), &count, EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA-256 calculation failed");
  std::ostringstream result;
  for (unsigned int i = 0; i < count; ++i) result << std::hex << std::setw(2) << std::setfill('0') << int(digest[i]);
  return result.str();
}
}  // namespace rgbd_object_detector
