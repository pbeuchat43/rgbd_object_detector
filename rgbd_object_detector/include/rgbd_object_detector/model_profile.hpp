// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rgbd_object_detector {
// The supported decoder is deliberately narrow: static YOLOv8 detection,
// batch one, RGB NCHW float32 input and channel-major xywh + class scores.
inline constexpr int kModelWidth = 640, kModelHeight = 640, kPredictions = 8400;
inline constexpr size_t kInputElements = 3 * kModelWidth * kModelHeight;

struct ModelProfile {
  std::string model_id, onnx_sha256, profile_sha256;
  std::string input_name, output_name, resize_mode;
  std::vector<std::string> classes;
  int padding_value = 114;
  double score_threshold = 0.5, iou_threshold = 0.5;
  int pre_nms_limit = 1000, max_detections = 20;
  size_t output_elements() const { return (4 + classes.size()) * kPredictions; }
  std::string preprocessing() const;
  bool accepts_legacy_manifest() const;
  void validate() const;
};
ModelProfile load_model_profile(const std::string& path);

struct ImageTransform {
  int resized_width = 0, resized_height = 0, left = 0, top = 0;
  // Continuous model edge coordinates -> continuous source edge coordinates.
  double scale_x = 0, scale_y = 0, offset_x = 0, offset_y = 0;
};
ImageTransform image_transform(const ModelProfile& profile, uint32_t width, uint32_t height);
}  // namespace rgbd_object_detector
