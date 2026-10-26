// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#pragma once
#include "rgbd_object_detector/core.hpp"
#include <rclcpp/rclcpp.hpp>
#include <functional>

namespace rgbd_object_detector {
using BackendFactory = std::function<std::unique_ptr<InferenceBackend>(const std::string&, const ModelProfile&)>;
class DetectorNode : public rclcpp::Node {
 public:
  explicit DetectorNode(const rclcpp::NodeOptions& options);
  // Tests inject a backend without opening a GPU or weakening production parameters.
  DetectorNode(const rclcpp::NodeOptions& options, BackendFactory factory);
  ~DetectorNode() override;
 private:
  class Implementation;
  std::unique_ptr<Implementation> impl_;
};
}  // namespace rgbd_object_detector
