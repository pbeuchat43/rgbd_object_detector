// Synthetic exact-source rendering and bounded presentation-worker checks.
#include "rgbd_object_detector/overlay.hpp"
#include "rgbd_object_detector/core.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
using namespace rgbd_object_detector;
using namespace std::chrono_literals;
using Frame = OverlayWorker::Frame;
using Batch = OverlayWorker::Batch;
using Overlay = rgbd_object_detector_interfaces::msg::OverlayFrame;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
std::pair<Frame::SharedPtr, Batch::SharedPtr> fixture(uint64_t sequence, const std::string& generation = "A") {
  auto frame = std::make_shared<Frame>(); auto batch = std::make_shared<Batch>();
  frame->capture.source_id = "fixture"; frame->capture.generation = generation; frame->capture.stream_id = "rgbd";
  frame->capture.sequence = sequence; frame->capture.acquisition_ns = monotonic_ns() - 1000000;
  frame->capture.received_ns = monotonic_ns(); frame->capture.clock_domain = local_clock_domain();
  frame->capture.unix_offset_ns = 1700000000000000000LL;
  frame->calibration_id = "fixture_calibration";
  auto& image = frame->rgb;
  image.header.frame_id = "optical";
  image.header.stamp = rclcpp::Time(frame->capture.acquisition_ns + frame->capture.unix_offset_ns);
  image.width = 640; image.height = 480; image.step = 640*3; image.encoding = "bgr8";
  image.data.assign(image.step*image.height, 100);
  batch->capture = frame->capture; batch->calibration_id = frame->calibration_id;
  batch->source_width = 640; batch->source_height = 480; batch->model_id = "fixture_cones";
  batch->max_frame_age_ns = 2000000000;
  batch->detections.header = image.header;
  vision_msgs::msg::Detection2D detection;
  detection.header = image.header; detection.bbox.center.position.x = 320; detection.bbox.center.position.y = 240;
  detection.bbox.size_x = detection.bbox.size_y = 160;
  vision_msgs::msg::ObjectHypothesisWithPose result;
  result.hypothesis.class_id = "bottle"; result.hypothesis.score = .75;
  detection.results.push_back(result); batch->detections.detections.push_back(detection);
  rgbd_object_detector_interfaces::msg::Spatial spatial;
  spatial.valid = false; spatial.invalid_reason = "insufficient_depth"; spatial.position.x = spatial.position.y = spatial.position.z = NAN;
  batch->spatial.push_back(spatial);
  return {frame, batch};
}
void rendering() {
  const auto [frame, batch] = fixture(1234);
  const auto original = frame->rgb;
  auto output = render_overlay(frame->rgb, *batch, true, true);
  check(frame->rgb == original && output->header == original.header && output->step == original.step,
        "rendering mutated source or acquisition metadata");
  // Probe the middle of the left edge; the class label covers the top corner.
  const auto border = 240*output->step + 240*3;
  check(output->data[border] == 0 && output->data[border+1] == 210 && output->data[border+2] == 255,
        "invalid spatial result lost its visible 2D box");
  std::array<uint8_t,15> bytes{};
  for (int bit = 0; bit < 120; ++bit) {
    const int x = 8 + (bit%60)*8 + 4, y = 480 - 24 + (bit/60)*8 + 4;
    if (output->data[y*output->step + x*3] > 127) bytes[bit/8] |= 1 << (bit%8);
  }
  uint64_t stamp = 0; uint32_t sequence = 0;
  for (int i = 0; i < 8; ++i) stamp |= uint64_t(bytes[1+i]) << (8*i);
  for (int i = 0; i < 4; ++i) sequence |= uint32_t(bytes[9+i]) << (8*i);
  check(bytes[0] == 0xA5 && sequence == 1234 && stamp == uint64_t(rclcpp::Time(original.header.stamp).nanoseconds()),
        "measurement marker lost capture identity/acquisition time");
  batch->detections.detections.clear(); batch->spatial.clear();
  output = render_overlay(frame->rgb, *batch, false, false);
  check(output->data[border] == 100, "empty result retained previous overlay pixels");
  batch->detections.header.frame_id = "unrelated";
  bool rejected = false;
  try { render_overlay(frame->rgb, *batch, false, false); } catch (const std::exception&) { rejected = true; }
  check(rejected, "unrelated image header accepted");
}
void worker() {
  auto node = std::make_shared<rclcpp::Node>("overlay_fixture", rclcpp::NodeOptions().use_intra_process_comms(true));
  rclcpp::executors::SingleThreadedExecutor executor; executor.add_node(node);
  std::vector<Overlay::ConstSharedPtr> images;
  auto subscription = node->create_subscription<Overlay>("/overlay_fixture/overlay/frame", rclcpp::SensorDataQoS().keep_last(1),
      [&](Overlay::ConstSharedPtr message) { images.push_back(message); });
  std::atomic<bool> held{true}; std::atomic<int> entered{0};
  OverlayWorker output(*node, true, false, [&](const auto& image, const auto& batch, bool spatial, bool capture_marker) {
    ++entered;
    while (held) std::this_thread::sleep_for(1ms);
    return render_overlay(image, batch, spatial, capture_marker);
  });
  struct ReleaseHold { std::atomic<bool>& held; ~ReleaseHold() { held = false; } } release{held};
  auto until = [&](auto condition, const char* error, bool spin = true) {
    const auto deadline = monotonic_ns() + 2000000000;
    while (!condition() && monotonic_ns() < deadline) {
      if (spin) executor.spin_some();
      std::this_thread::sleep_for(1ms);
    }
    if (!condition()) { held = false; throw std::runtime_error(error); }
  };
  output.invalidate(1);
  auto [first, batch] = fixture(1); output.submit(first, batch, 1);
  until([&] { return entered == 1; }, "presentation worker did not begin");
  for (int sequence = 2; sequence <= 10; ++sequence) {
    auto pair = fixture(sequence); output.submit(pair.first, pair.second, 1);
  }
  check(output.statistics().pending == 1 && output.statistics().replaced == 8, "overlay work is not bounded/latest-only");
  output.invalidate(2);
  auto latest = fixture(11, "B"); output.submit(latest.first, latest.second, 2);
  held = false;
  until([&] { return images.size() == 1; }, "new-generation overlay missing");
  check(images[0]->batch.capture.generation == "B" && images[0]->batch.capture.sequence == 11 &&
        output.statistics().superseded == 2, "old-generation overlay escaped invalidation");
  // Stop servicing the subscriber. The worker must keep publishing/replacing
  // independently of reader callback progress, with no retained FIFO growth.
  for (int sequence = 12; sequence <= 16; ++sequence) {
    auto pair = fixture(sequence, "B"); output.submit(pair.first, pair.second, 2);
    until([&] { return output.statistics().rendered == uint64_t(sequence-10); }, "slow reader stalled rendering", false);
  }
  until([&] { return images.size() == 2; }, "latest overlay not delivered after slow reader resumed");
  check(images.back()->batch.capture.sequence == 16 && output.statistics().pending == 0,
        "slow reader retained old overlay backlog");
  auto stale = fixture(17, "B"); stale.second->max_frame_age_ns = 1;
  output.submit(stale.first, stale.second, 2);
  until([&] { return output.statistics().stale == 1; }, "stale overlay was not dropped");
  auto wrong = fixture(18, "B"); wrong.second->capture.sequence = 999;
  output.submit(wrong.first, wrong.second, 2);
  until([&] { return output.statistics().failures == 1; }, "overlay capture mismatch did not fail closed");
  check(output.statistics().pending == 0 && !output.statistics().processing, "failed overlay worker retained work");
}
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rendering(); worker(); rclcpp::shutdown();
    std::cout << "overlay checks passed (synthetic captures, bounded worker, real ROS subscriber)\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; rclcpp::shutdown(); return 1; }
}
