// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#include "rgbd_object_detector/core.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>

namespace rgbd_object_detector {
namespace {
void require(bool value, const char* message) {
  if (!value) throw std::invalid_argument(message);
}
bool digest(const std::string& value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
void tensor(const cv::FileNode& node, const std::vector<int>& shape) {
  require(node.isMap() && std::string(node["dtype"]) == "float32", "profile requires float32 tensors");
  const auto actual = node["shape"];
  require(actual.isSeq() && actual.size() == shape.size(), "unsupported profile tensor rank");
  size_t index = 0;
  for (const auto& dimension : actual) {
    require(dimension.isInt() && int(dimension) == shape.at(index++), "unsupported profile tensor dimension");
  }
}
}
void ModelProfile::validate() const {
  require(!model_id.empty() && model_id.size() <= 128, "invalid model ID");
  require(digest(onnx_sha256) && digest(profile_sha256), "profile requires lowercase SHA-256 identities");
  require(!input_name.empty() && input_name.size() <= 128 && !output_name.empty() &&
          output_name.size() <= 128 && input_name != output_name, "invalid tensor names");
  require(resize_mode == "stretch" || resize_mode == "letterbox", "unsupported resize mode");
  require(padding_value >= 0 && padding_value <= 255, "invalid letterbox padding value");
  require(!classes.empty() && classes.size() <= 256, "profile requires 1..256 classes");
  Config defaults;
  defaults.score_threshold = score_threshold; defaults.iou_threshold = iou_threshold;
  defaults.pre_nms_limit = pre_nms_limit; defaults.max_detections = max_detections;
  defaults.validate();
  std::set<std::string> unique;
  for (const auto& name : classes) {
    require(!name.empty() && name.size() <= 80 && std::all_of(name.begin(), name.end(), [](unsigned char c) {
      return c >= 32 && c <= 126;
    }) && unique.insert(name).second, "class names must be unique printable ASCII, 1..80 bytes");
  }
}
std::string ModelProfile::preprocessing() const {
  return "bgr8_to_rgb_nchw_div255_bilinear_" + resize_mode;
}
bool ModelProfile::accepts_legacy_manifest() const {
  // The only pre-profile engine contract. Do not extend this compatibility gate
  // to unrelated weights, class maps, tensor names or preprocessing.
  return model_id == "cones-2024-standard-20260927" &&
    onnx_sha256 == "dcf4d911dee975944a3cca895d92803c3181d8dba670759276c4d71901c8040d" &&
    input_name == "images" && output_name == "output0" && resize_mode == "stretch" &&
    classes == std::vector<std::string>{"yellow_cone", "blue_cone"};
}
ModelProfile load_model_profile(const std::string& path) {
  std::ifstream source(path, std::ios::binary | std::ios::ate);
  require(bool(source), "cannot open model profile");
  const auto size = source.tellg();
  require(size > 0 && size <= 65536, "model profile must be 1..65536 bytes");
  std::string bytes(size_t(size), '\0');
  source.seekg(0); source.read(bytes.data(), bytes.size());
  require(bool(source), "cannot read model profile");
  cv::FileStorage file(bytes, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
  require(file.isOpened() && int(file["schema_version"]) == 2, "unsupported model profile schema");
  ModelProfile profile;
  profile.model_id = std::string(file["model_id"]);
  profile.onnx_sha256 = std::string(file["onnx_sha256"]);
  profile.profile_sha256 = sha256_bytes(bytes.data(), bytes.size());
  profile.input_name = std::string(file["input"]["name"]);
  profile.output_name = std::string(file["output"]["name"]);
  require(file["classes"].isSeq(), "profile class catalogue must be a sequence");
  for (const auto& name : file["classes"]) {
    require(name.isString(), "class names must be strings");
    profile.classes.push_back(std::string(name));
  }
  profile.resize_mode = std::string(file["resize"]["mode"]);
  require(file["resize"]["padding_value"].isInt(), "profile must declare padding value");
  profile.padding_value = int(file["resize"]["padding_value"]);
  const auto defaults = file["defaults"];
  require(defaults.isMap() && (defaults["score_threshold"].isReal() || defaults["score_threshold"].isInt()) &&
          (defaults["iou_threshold"].isReal() || defaults["iou_threshold"].isInt()) &&
          defaults["pre_nms_limit"].isInt() && defaults["max_detections"].isInt(), "invalid profile detection defaults");
  profile.score_threshold = double(defaults["score_threshold"]); profile.iou_threshold = double(defaults["iou_threshold"]);
  profile.pre_nms_limit = int(defaults["pre_nms_limit"]); profile.max_detections = int(defaults["max_detections"]);
  require(std::string(file["preprocessing"]) == "bgr8_to_rgb_nchw_div255_bilinear" &&
          std::string(file["decoder"]) == "yolov8_xywh_class_scores", "unsupported preprocessing/decoder contract");
  profile.validate();
  tensor(file["input"], {1, 3, kModelHeight, kModelWidth});
  tensor(file["output"], {1, int(4 + profile.classes.size()), kPredictions});
  return profile;
}
ImageTransform image_transform(const ModelProfile& profile, uint32_t width, uint32_t height) {
  require(width > 0 && width <= 1920 && height > 0 && height <= 1080, "invalid source dimensions");
  ImageTransform transform;
  transform.resized_width = kModelWidth; transform.resized_height = kModelHeight;
  if (profile.resize_mode == "letterbox") {
    const double gain = std::min(double(kModelWidth) / width, double(kModelHeight) / height);
    // Nearest-even agrees with Python round in the pinned reference LetterBox.
    transform.resized_width = int(std::nearbyint(width * gain));
    transform.resized_height = int(std::nearbyint(height * gain));
    require(transform.resized_width > 0 && transform.resized_height > 0, "source aspect ratio is too extreme for letterboxing");
    transform.left = (kModelWidth - transform.resized_width) / 2;
    transform.top = (kModelHeight - transform.resized_height) / 2;
    // YOLO's continuous-coordinate inverse uses the nominal uniform gain.
    transform.scale_x = transform.scale_y = 1.0 / gain;
  } else {
    require(profile.resize_mode == "stretch", "unsupported resize mode");
    transform.scale_x = double(width) / kModelWidth;
    transform.scale_y = double(height) / kModelHeight;
  }
  transform.offset_x = -transform.left * transform.scale_x;
  transform.offset_y = -transform.top * transform.scale_y;
  return transform;
}
}  // namespace rgbd_object_detector
