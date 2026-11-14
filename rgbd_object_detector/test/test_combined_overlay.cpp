// Copyright (c) 2026 Paul N. Beuchat. MIT License.
// Synthetic inputs through actual ROS subscriptions; rendering can be held.
#include "rgbd_object_detector/combined_overlay.hpp"
#include "rgbd_object_detector/core.hpp"
#include <chrono>
#include <iostream>
#include <thread>
using namespace rgbd_object_detector;
using namespace std::chrono_literals;
using Frame = CombinedOverlayNode::Frame;
using CaptureDetections = CombinedOverlayNode::CaptureDetections;
using Output = rgbd_object_detector_interfaces::msg::CombinedOverlayFrame;
void check(bool condition, const char* text) { if (!condition) throw std::runtime_error(text); }
struct Sample {
  Frame::SharedPtr frame = std::make_shared<Frame>();
  std::array<CaptureDetections::SharedPtr, 2> capture_detections{std::make_shared<CaptureDetections>(), std::make_shared<CaptureDetections>()};
  Sample(uint64_t sequence, const std::string& generation = "A") {
    auto& c = frame->capture;
    c.source_id = "fixture"; c.generation = generation; c.stream_id = "rgbd"; c.sequence = sequence;
    c.acquisition_ns = monotonic_ns() - 1000000; c.received_ns = monotonic_ns(); c.clock_domain = local_clock_domain();
    c.unix_offset_ns = 1700000000000000000LL; frame->calibration_id = "fixture-calibration";
    auto& image = frame->rgb; image.header.frame_id = "optical";
    image.header.stamp = rclcpp::Time(c.acquisition_ns + c.unix_offset_ns);
    image.width = 640; image.height = 480; image.step = 640*3 + 12; image.encoding = "bgr8";
    image.data.assign(image.step*image.height, 100);
    for (size_t i = 0; i < 2; ++i) {
      auto& b = *capture_detections[i]; b.capture = c; b.calibration_id = frame->calibration_id;
      b.source_width = image.width; b.source_height = image.height;
      b.model_id = i == 0 ? "cones" : "coco"; b.max_frame_age_ns = 2000000000;
      b.detections.header = image.header;
      vision_msgs::msg::Detection2D detection; detection.header = image.header;
      detection.bbox.center.position.x = 180 + i*260; detection.bbox.center.position.y = 240;
      detection.bbox.size_x = 80; detection.bbox.size_y = 160;
      vision_msgs::msg::ObjectHypothesisWithPose result; result.hypothesis.class_id = "object"; result.hypothesis.score = .75;
      detection.results.push_back(result); b.detections.detections.push_back(detection); b.spatial.emplace_back();
    }
  }
};
void rendering() {
  Sample sample(42); const auto original = sample.frame->rgb;
  const auto image = render_combined_overlay(original, *sample.capture_detections[0], *sample.capture_detections[1], true, false);
  check(sample.frame->rgb == original && image->header == original.header && image->step == original.step,
        "combined rendering changed source/storage/acquisition metadata");
  const auto first = 240*image->step + 140*3, second = 240*image->step + 400*3;
  check(image->data[first] == 0 && image->data[first+1] == 210 && image->data[first+2] == 255 &&
        image->data[second] == 255 && image->data[second+1] == 210 && image->data[second+2] == 80,
        "model colours or invalid-depth boxes missing");
  sample.capture_detections[0]->detections.detections.clear(); sample.capture_detections[0]->spatial.clear();
  const auto empty = render_combined_overlay(original, *sample.capture_detections[0], *sample.capture_detections[1], false, false);
  check(empty->data[first] == 100 && empty->data[second] == 255, "valid empty result retained old detections");
  sample.capture_detections[1]->depth_sequence++;
  bool rejected = false;
  try { render_combined_overlay(original, *sample.capture_detections[0], *sample.capture_detections[1], false, false); }
  catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, "incompatible pairing metadata accepted");
  sample.capture_detections[1]->depth_sequence--;
  sample.capture_detections[1]->depth_exposure_midpoint_ns = 123;
  rejected = false;
  try { render_combined_overlay(original, *sample.capture_detections[0], *sample.capture_detections[1], false, false); }
  catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, "different exposure metadata accepted");
}
struct Harness {
  std::atomic<bool> held{false}; std::atomic<int> entered{0};
  std::shared_ptr<CombinedOverlayNode> output;
  std::shared_ptr<rclcpp::Node> producer;
  rclcpp::executors::SingleThreadedExecutor executor;
  rclcpp::Publisher<Frame>::SharedPtr frames;
  std::array<rclcpp::Publisher<CaptureDetections>::SharedPtr, 2> capture_detections;
  rclcpp::Subscription<Output>::SharedPtr subscription;
  std::vector<Output::ConstSharedPtr> images;
  Harness() {
    auto options = rclcpp::NodeOptions().use_intra_process_comms(true);
    options.arguments({"--ros-args", "-r", "__ns:=/capture_rename",
        "-r", "frames:=camera/frames",
        "-r", "first_capture_detections:=detector_cones/capture_detections",
        "-r", "second_capture_detections:=detector_coco/capture_detections"});
    options.parameter_overrides({rclcpp::Parameter("join_capacity", 4), rclcpp::Parameter("max_frame_age_ms", 2000)});
    output = std::make_shared<CombinedOverlayNode>(options, [this](const auto& image, const auto& a, const auto& b, bool spatial, bool marker) {
      ++entered; while (held) std::this_thread::sleep_for(1ms);
      return render_combined_overlay(image, a, b, spatial, marker);
    });
    producer = std::make_shared<rclcpp::Node>("combined_fixture", rclcpp::NodeOptions().use_intra_process_comms(true));
    const auto qos = rclcpp::SensorDataQoS().keep_last(1);
    frames = producer->create_publisher<Frame>("/capture_rename/camera/frames", qos);
    capture_detections[0] = producer->create_publisher<CaptureDetections>("/capture_rename/detector_cones/capture_detections", qos);
    capture_detections[1] = producer->create_publisher<CaptureDetections>("/capture_rename/detector_coco/capture_detections", qos);
    subscription = producer->create_subscription<Output>("/capture_rename/combined_overlay/overlay/frame", qos,
        [this](Output::ConstSharedPtr image) { images.push_back(std::move(image)); });
    executor.add_node(output); executor.add_node(producer);
  }
  ~Harness() { held = false; }
  template<class Condition> void until(Condition condition, const char* message) {
    const auto deadline = monotonic_ns() + 3000000000LL;
    while (!condition() && monotonic_ns() < deadline) { executor.spin_some(); std::this_thread::sleep_for(1ms); }
    if (!condition()) { held = false; throw std::runtime_error(message); }
  }
  void frame(const Sample& s) {
    const auto before = output->statistics().received[0]; frames->publish(*s.frame);
    until([&] { return output->statistics().received[0] > before; }, "frame callback blocked");
  }
  void publish_capture_detections(const Sample& s, size_t i) {
    const auto before = output->statistics().received[i+1]; capture_detections[i]->publish(*s.capture_detections[i]);
    until([&] { return output->statistics().received[i+1] > before; }, "capture_detections callback blocked");
  }
  void complete(const Sample& s) { frame(s); publish_capture_detections(s, 1); publish_capture_detections(s, 0); }
};
void joining() {
  Harness h;
  Sample first(1), next(2);
  h.frame(first); h.frame(next); h.publish_capture_detections(first, 1); h.publish_capture_detections(first, 0);
  h.until([&] { return h.images.size() == 1; }, "shuffled streams failed to join retained older capture");
  check(h.images.back()->capture_detections[0] == *first.capture_detections[0] && h.images.back()->capture_detections[1] == *first.capture_detections[1],
        "combined output changed model provenance");
  h.publish_capture_detections(next, 0);
  for (uint64_t n = 3; n < 15; ++n) { Sample s(n); h.frame(s); h.publish_capture_detections(s, 0); }
  check(h.output->statistics().retained <= 4 && h.output->statistics().evicted >= 9 && h.output->statistics().matched == 1,
        "missing second detector grew cache or produced partial overlay");
  h.publish_capture_detections(next, 1); // A late second result must not combine with a newer first result.
  check(h.output->statistics().matched == 1, "cross-sequence join");
  Sample bad(15); bad.capture_detections[1]->capture.received_ns--;
  h.complete(bad);
  check(h.output->statistics().matched == 1 && h.output->statistics().rejected > 0, "partial capture identity accepted");
  Sample mismatch(16); mismatch.capture_detections[1]->depth_sequence++;
  h.complete(mismatch);
  check(h.output->statistics().matched == 1, "frame/result pairing mismatch accepted");
  Sample good(17); good.capture_detections[0]->detections.detections.clear(); good.capture_detections[0]->spatial.clear();
  h.complete(good);
  h.until([&] { return h.images.size() == 2; }, "valid empty first result did not complete pair");
  h.subscription.reset(); Sample idle(18); h.complete(idle);
  check(h.output->statistics().retained == 0 && h.output->statistics().pending == 0 && h.output->statistics().no_consumers > 0,
        "no-consumer input retained work");
}
void worker() {
  Harness h; h.held = true; Sample first(1); h.complete(first);
  h.until([&] { return h.entered == 1; }, "renderer did not start");
  for (uint64_t n = 2; n <= 6; ++n) { Sample s(n); h.complete(s); }
  auto stats = h.output->statistics();
  check(stats.pending == 1 && stats.replaced == 4 && stats.received[1] == 6 && stats.received[2] == 6,
        "slow rendering blocked inputs or grew render queue");
  Sample changed(1, "B"); h.complete(changed);
  h.publish_capture_detections(first, 1); // Old generation must not roll state back.
  h.held = false;
  h.until([&] { return h.images.size() == 1; }, "new generation missing");
  check(h.images[0]->capture_detections[0].capture.generation == "B" && h.output->statistics().generation == "B",
        "old generation published or rolled back current generation");
  // A late render is discarded using the stricter of the two model age limits.
  h.held = true; const auto entered = h.entered.load(); const auto expired = h.output->statistics().expired;
  Sample stale(2, "B"); stale.capture_detections[1]->max_frame_age_ns = 50000000;
  h.complete(stale); h.until([&] { return h.entered > entered; }, "stale fixture did not begin rendering");
  std::this_thread::sleep_for(65ms); h.held = false;
  h.until([&] { return h.output->statistics().expired > expired; }, "late render was not expired");
  check(h.images.size() == 1, "expired image published");
  // Expire an incomplete join without needing another camera callback.
  Sample missing(3, "B"); missing.capture_detections[0]->max_frame_age_ns = 10000000;
  h.frame(missing); h.publish_capture_detections(missing, 0);
  h.until([&] { return h.output->statistics().retained == 0; }, "incomplete join did not expire");
  Sample short_age(4, "B"); short_age.capture_detections[0]->max_frame_age_ns = 80000000;
  h.complete(short_age);
  h.until([&] { return h.output->statistics().rendered == 2; }, "short-age complete result missing");
  check(h.output->statistics().state == "streaming", "fresh result not streaming");
  h.until([&] { return h.output->statistics().state == "waiting_for_pair"; }, "status ignored stricter model age limit");
}
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rendering(); joining(); worker(); rclcpp::shutdown();
    std::cout << "combined overlay checks passed (synthetic inputs and real ROS subscriptions)\n"; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; rclcpp::shutdown(); return 1; }
}
