// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#pragma once
#include "rgbd_object_detector/overlay.hpp"
#include <rgbd_object_detector_interfaces/msg/combined_overlay_frame.hpp>
#include <rgbd_object_detector_interfaces/msg/combined_overlay_status.hpp>
#include <array>
#include <map>

namespace rgbd_object_detector {
// Presentation only: no camera session, inference backend or GPU ownership.
class CombinedOverlayNode : public rclcpp::Node {
 public:
  using Frame = rgbd_camera_interfaces::msg::Frame;
  using Batch = rgbd_object_detector_interfaces::msg::Batch;
  using Status = rgbd_object_detector_interfaces::msg::CombinedOverlayStatus;
  using Render = std::function<std::unique_ptr<sensor_msgs::msg::Image>(
      const sensor_msgs::msg::Image&, const Batch&, const Batch&, bool, bool)>;
  explicit CombinedOverlayNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions(),
                               Render renderer = render_combined_overlay);
  ~CombinedOverlayNode() override;
  Status statistics();
 private:
  struct Entry {
    Frame::ConstSharedPtr frame;
    std::array<Batch::ConstSharedPtr, 2> batches;
    int64_t acquisition_ns = 0;
  };
  struct Job { Entry entry; uint64_t epoch; };
  void receive_frame(Frame::ConstSharedPtr frame);
  void receive_batch(size_t index, Batch::ConstSharedPtr batch);
  bool capture_valid(const rgbd_camera_interfaces::msg::CaptureReference& capture, int64_t now) const;
  bool has_consumers() const;
  bool fresh(const Entry& entry, int64_t now) const;
  void expire(int64_t now);
  void join(uint64_t sequence);
  void clear();
  void run();
  void publish_status();
  bool spatial_, marker_, have_frame_ = false, have_match_ = false;
  size_t capacity_;
  int64_t max_age_ns_, last_frame_acquisition_ = 0;
  uint64_t last_frame_sequence_ = 0, last_matched_sequence_ = 0;
  std::array<uint64_t, 2> last_batch_sequence_{};
  std::array<bool, 2> have_batch_{};
  std::array<std::string, 2> model_identity_;
  std::string calibration_;
  Render renderer_;
  std::atomic<bool> stopping_{false};
  std::atomic<uint64_t> epoch_{0};
  std::mutex mutex_;
  std::condition_variable available_;
  std::map<uint64_t, Entry> entries_;
  std::unique_ptr<Job> pending_;
  Status status_;
  rclcpp::Subscription<Frame>::SharedPtr input_;
  std::array<rclcpp::Subscription<Batch>::SharedPtr, 2> batches_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr images_;
  rclcpp::Publisher<rgbd_object_detector_interfaces::msg::CombinedOverlayFrame>::SharedPtr frames_;
  rclcpp::Publisher<Status>::SharedPtr statuses_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::thread worker_;
};
}  // namespace rgbd_object_detector
