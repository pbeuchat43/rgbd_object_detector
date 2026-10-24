// Copyright (c) 2026 Paul N. Beuchat. MIT License. See NOTICE.md.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rgbd_object_detector {
inline constexpr int kModelWidth = 640, kModelHeight = 640, kPredictions = 8400;
inline constexpr size_t kInputElements = 3 * kModelWidth * kModelHeight;
inline constexpr size_t kOutputElements = 6 * kPredictions;
inline constexpr const char* kModelId = "cones-2024-standard-20260927";
inline constexpr const char* kModelSha256 = "dcf4d911dee975944a3cca895d92803c3181d8dba670759276c4d71901c8040d";
inline constexpr std::array<const char*, 2> kClassNames{"yellow_cone", "blue_cone"};
inline constexpr const char* kSpatialMethod = "central_roi_median_z_box_center";

struct Config {
  double score_threshold = 0.5, iou_threshold = 0.5;
  int pre_nms_limit = 1000, max_detections = 20;
  double roi_fraction = 0.5, min_depth_m = 0.1, max_depth_m = 5.0;
  int min_valid_pixels = 5;
  double min_valid_fraction = 0.05, max_frame_age_ms = 500.0;
  void validate() const;
};

struct ImageView {
  const uint8_t* data = nullptr;
  size_t bytes = 0;
  uint32_t width = 0, height = 0, step = 0;
  void validate(uint32_t channels) const;
};
struct DepthView {
  ImageView image;
  std::array<double, 9> k{};
  double unit_m = 0.001;
  uint16_t invalid = 0;
  void validate() const;
};
struct Box {
  double x1 = 0, y1 = 0, x2 = 0, y2 = 0, score = 0;
  uint32_t class_index = 0, prediction_index = 0;
};
struct Decoded {
  std::vector<Box> boxes;
  uint32_t candidate_count = 0, nms_candidate_count = 0, nms_count = 0;
  bool truncated = false;
};
struct SpatialEstimate {
  bool valid = false;
  std::string reason;
  std::array<double, 3> xyz;
  double u = 0, v = 0, valid_fraction = 0, depth_mad_m;
  uint32_t roi_pixels = 0, valid_pixels = 0;
  SpatialEstimate();
};

// Writes exactly kInputElements floats. The source storage is never modified.
void preprocess(const ImageView& bgr, float* destination, size_t elements);
Decoded decode(const float* output, size_t elements, uint32_t source_width,
               uint32_t source_height, const Config& config);
SpatialEstimate estimate(const Box& box, const DepthView& depth, const Config& config);
int64_t monotonic_ns();
std::string local_clock_domain();
std::string sha256_file(const std::string& path);
std::string sha256_bytes(const void* data, size_t bytes);

struct BackendIdentity {
  std::string name;
  std::string model_id = kModelId, model_sha256 = kModelSha256, engine_sha256;
};
// A single worker owns this object and its buffers. infer() completes transfers
// before returning; returned output remains valid until the next infer/destruction.
class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;
  virtual float* input() = 0;
  virtual const float* infer() = 0;
  virtual BackendIdentity identity() const = 0;
};
std::unique_ptr<InferenceBackend> make_tensorrt_backend(const std::string& engine_path);
}  // namespace rgbd_object_detector
