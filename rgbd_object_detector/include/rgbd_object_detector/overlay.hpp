// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#pragma once
#include <rgbd_camera_interfaces/msg/frame.hpp>
#include <rgbd_object_detector_interfaces/msg/capture_detections.hpp>
#include <rgbd_object_detector_interfaces/msg/overlay_frame.hpp>
#include <rclcpp/rclcpp.hpp>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace rgbd_object_detector {
struct OverlayStatistics {
  bool processing = false;
  uint32_t pending = 0;
  uint64_t rendered = 0, replaced = 0, stale = 0, superseded = 0, no_consumers = 0, failures = 0;
  int64_t last_acquisition_ns = 0, last_render_ns = 0;
  std::string error;
};
// Presentation code is independent of the inference/geometry library. The
// returned pixels own separate storage; source and result are never modified.
std::unique_ptr<sensor_msgs::msg::Image> render_overlay(
    const sensor_msgs::msg::Image& source, const rgbd_object_detector_interfaces::msg::CaptureDetections& capture_detections,
    bool spatial, bool capture_marker);
std::unique_ptr<sensor_msgs::msg::Image> render_combined_overlay(
    const sensor_msgs::msg::Image& source, const rgbd_object_detector_interfaces::msg::CaptureDetections& first,
    const rgbd_object_detector_interfaces::msg::CaptureDetections& second, bool spatial, bool capture_marker);

class OverlayWorker {
 public:
  using Frame = rgbd_camera_interfaces::msg::Frame;
  using CaptureDetections = rgbd_object_detector_interfaces::msg::CaptureDetections;
  using Render = std::function<std::unique_ptr<sensor_msgs::msg::Image>(
      const sensor_msgs::msg::Image&, const CaptureDetections&, bool, bool)>;
  OverlayWorker(rclcpp::Node& node, bool spatial, bool capture_marker,
                Render renderer = render_overlay);
  ~OverlayWorker();
  void submit(Frame::ConstSharedPtr frame, CaptureDetections::ConstSharedPtr capture_detections, uint64_t epoch);
  void invalidate(uint64_t epoch);
  OverlayStatistics statistics();
 private:
  struct Job { Frame::ConstSharedPtr frame; CaptureDetections::ConstSharedPtr capture_detections; uint64_t epoch; };
  void run();
  bool has_consumers() const;
  bool spatial_, capture_marker_;
  Render renderer_;
  std::atomic<bool> stopping_{false};
  std::atomic<uint64_t> epoch_{0};
  std::mutex mutex_;
  std::condition_variable available_;
  std::unique_ptr<Job> pending_;
  OverlayStatistics statistics_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr images_;
  rclcpp::Publisher<rgbd_object_detector_interfaces::msg::OverlayFrame>::SharedPtr frames_;
  std::thread worker_;
};
}  // namespace rgbd_object_detector
