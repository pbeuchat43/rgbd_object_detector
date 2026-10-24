// Synthetic backend in a real intra-process ROS graph. No camera/GPU is opened.
#include "rgbd_object_detector/detector_node.hpp"
#include <rgbd_camera_interfaces/msg/frame.hpp>
#include <rgbd_object_detector_interfaces/msg/batch.hpp>
#include <rgbd_object_detector_interfaces/msg/status.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <thread>

using namespace rgbd_object_detector;
using namespace std::chrono_literals;
using Frame = rgbd_camera_interfaces::msg::Frame;
using Batch = rgbd_object_detector_interfaces::msg::Batch;
using Status = rgbd_object_detector_interfaces::msg::Status;
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

struct Control {
  std::atomic<bool> hold{false}, fail{false}, destroyed{false};
  std::atomic<int> entered{0}, delay_ms{0};
};
class FixtureBackend : public InferenceBackend {
 public:
  explicit FixtureBackend(std::shared_ptr<Control> control) : control_(std::move(control)), input_(kInputElements), output_(kOutputElements) {}
  ~FixtureBackend() override { control_->destroyed = true; }
  float* input() override { return input_.data(); }
  const float* infer() override {
    ++control_->entered;
    while (control_->hold) std::this_thread::sleep_for(1ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(control_->delay_ms.load()));
    if (control_->fail) throw std::runtime_error("fixture_backend_failure");
    std::fill(output_.begin(), output_.end(), 0);
    if (input_[0] > 0) {
      const float box[]{320,320,160,160,.9F,.1F};
      for (int channel = 0; channel < 6; ++channel) output_[channel*kPredictions] = box[channel];
    }
    return output_.data();
  }
  BackendIdentity identity() const override { return {"synthetic_fixture", kModelId, kModelSha256, "fixture_engine"}; }
 private:
  std::shared_ptr<Control> control_;
  std::vector<float> input_, output_;
};

Frame fixture(uint64_t sequence, uint16_t depth = 2000, const std::string& generation = "A", bool empty = false) {
  Frame frame;
  const auto now = monotonic_ns();
  frame.capture.source_id = "fixture_camera"; frame.capture.generation = generation; frame.capture.stream_id = "rgbd";
  frame.capture.sequence = sequence; frame.capture.acquisition_ns = now - 1000000; frame.capture.received_ns = now;
  frame.capture.clock_domain = local_clock_domain(); frame.capture.unix_offset_ns = 1700000000000000000LL;
  frame.depth_sequence = sequence + 100; frame.depth_acquisition_ns = frame.capture.acquisition_ns + 100000;
  frame.skew_ns = 100000; frame.pairing_tolerance_ns = 5000000;
  frame.calibration_id = "calibration_" + generation; frame.backend = "fixture_camera";
  frame.alignment = "aligned_to_rgb"; frame.depth_convention = "optical_z"; frame.depth_unit_m = .001;
  frame.rgb.width = frame.depth.width = 320; frame.rgb.height = frame.depth.height = 240;
  frame.rgb.step = 320 * 3; frame.depth.step = 320 * 2;
  frame.rgb.encoding = "bgr8"; frame.depth.encoding = "16UC1";
  frame.rgb.header.frame_id = frame.depth.header.frame_id = "fixture_optical";
  frame.rgb.header.stamp = rclcpp::Time(frame.capture.acquisition_ns + frame.capture.unix_offset_ns);
  frame.depth.header.stamp = rclcpp::Time(frame.depth_acquisition_ns + frame.capture.unix_offset_ns);
  frame.rgb.data.resize(320*240*3, empty ? 0 : 100); frame.depth.data.resize(320*240*2);
  for (size_t index = 0; index < frame.depth.data.size(); index += 2) {
    frame.depth.data[index] = depth & 255; frame.depth.data[index+1] = depth >> 8;
  }
  auto& info = frame.rgb_info;
  info.header = frame.rgb.header; info.width = 320; info.height = 240;
  info.k = {200,0,160,0,200,120,0,0,1}; info.p = {200,0,160,0,0,200,120,0,0,0,1,0};
  info.r = {1,0,0,0,1,0,0,0,1}; info.d.assign(5,0); info.distortion_model = "plumb_bob";
  frame.depth_info = info; frame.depth_info.header = frame.depth.header;
  return frame;
}

class Harness {
 public:
  explicit Harness(double age = 500.0) : control(std::make_shared<Control>()) {
    auto options = rclcpp::NodeOptions().use_intra_process_comms(true)
      .parameter_overrides({{"engine_path", "fixture"}, {"max_frame_age_ms", age}});
    detector = std::make_shared<DetectorNode>(options, [state=control](const std::string&) { return std::make_unique<FixtureBackend>(state); });
    observer = std::make_shared<rclcpp::Node>("detector_fixture", rclcpp::NodeOptions().use_intra_process_comms(true));
    publisher = observer->create_publisher<Frame>("/frames", rclcpp::SensorDataQoS().keep_last(1));
    batch_sub = observer->create_subscription<Batch>("/rgbd_object_detector/batches", rclcpp::SensorDataQoS().keep_last(1),
      [this](Batch::ConstSharedPtr batch) { batches.push_back(batch); });
    status_sub = observer->create_subscription<Status>("/rgbd_object_detector/status", rclcpp::QoS(1).best_effort(),
      [this](Status::ConstSharedPtr state) { check(state->pending <= 1, "pending queue exceeded one"); status = state; });
    executor.add_node(detector); executor.add_node(observer);
    until([&] { return status && status->backend_ready; }, "backend did not become ready");
  }
  ~Harness() {
    control->hold = false;
    executor.remove_node(detector); executor.remove_node(observer); detector.reset();
  }
  void spin(int milliseconds) {
    const auto end = monotonic_ns() + milliseconds * 1000000LL;
    do { executor.spin_some(); std::this_thread::sleep_for(1ms); } while (monotonic_ns() < end);
  }
  void until(const std::function<bool()>& condition, const char* message, int milliseconds = 2000) {
    const auto end = monotonic_ns() + milliseconds * 1000000LL;
    while (!condition() && monotonic_ns() < end) spin(1);
    check(condition(), message);
  }
  void send(Frame frame) { publisher->publish(std::make_unique<Frame>(std::move(frame))); spin(3); }
  std::shared_ptr<Control> control;
  std::shared_ptr<DetectorNode> detector;
  std::vector<Batch::ConstSharedPtr> batches;
  Status::ConstSharedPtr status;
 private:
  std::shared_ptr<rclcpp::Node> observer;
  rclcpp::Publisher<Frame>::SharedPtr publisher;
  rclcpp::Subscription<Batch>::SharedPtr batch_sub;
  rclcpp::Subscription<Status>::SharedPtr status_sub;
  rclcpp::executors::SingleThreadedExecutor executor;
};

void output_and_validation() {
  Harness h;
  const auto input = fixture(1);
  h.send(input); h.until([&] { return h.batches.size() == 1; }, "no valid batch");
  const auto& batch = *h.batches[0];
  check(batch.capture == input.capture && batch.depth_sequence == 101 && batch.depth_acquisition_ns == input.depth_acquisition_ns &&
        batch.calibration_id == input.calibration_id && batch.detections.header == input.rgb.header, "source association lost");
  check(batch.detections.detections.size() == 1 && batch.spatial.size() == 1 && batch.spatial[0].valid &&
        batch.spatial[0].position.z == 2 && batch.spatial[0].position.x == 0 && batch.spatial[0].position.y == 0,
        "matched XYZ wrong");
  check(batch.model_to_source_x == .5 && batch.model_to_source_y == .375 &&
        batch.detections.detections[0].id.empty() && std::isnan(batch.detections.detections[0].results[0].pose.pose.position.x),
        "transform or unavailable tracking/pose semantics wrong");
  check(!h.detector->set_parameters({rclcpp::Parameter("score_threshold", .1)})[0].successful, "live reconfiguration accepted");
  h.send(fixture(2,0)); h.until([&] { return h.batches.size() == 2; }, "invalid depth lost 2D batch");
  check(h.batches[1]->detections.detections.size() == 1 && !h.batches[1]->spatial[0].valid &&
        h.batches[1]->spatial[0].invalid_reason == "insufficient_depth" && std::isnan(h.batches[1]->spatial[0].position.z),
        "invalid depth invented position or discarded 2D");
  h.send(fixture(3,2000,"A",true)); h.until([&] { return h.batches.size() == 3; }, "no empty batch");
  check(h.batches.back()->detections.detections.empty() && h.batches.back()->spatial.empty(), "empty batch not explicit");
  auto bad = fixture(4); bad.capture.clock_domain = "other_boot"; h.send(bad);
  bad = fixture(5); bad.skew_ns = 6000000; h.send(bad);
  bad = fixture(6); bad.rgb_info.k[0] = 100; h.send(bad);
  bad = fixture(7); bad.calibration_id = "changed_without_restart"; h.send(bad);
  h.send(fixture(3)); // Fresh timestamp, but regressed sequence in the same generation.
  h.until([&] { return h.status->rejected >= 5; }, "invalid input did not reject");
  check(h.batches.size() == 3, "invalid input reached inference");
  h.send(fixture(8)); h.until([&] { return h.batches.size() == 4; }, "valid input failed to recover");
}
void bounded_exact_association() {
  Harness h;
  h.control->hold = true; h.send(fixture(1,1010));
  h.until([&] { return h.control->entered == 1; }, "inference did not begin");
  for (uint64_t i = 2; i <= 15; ++i) h.send(fixture(i,1000+10*i));
  h.control->hold = false;
  h.until([&] { return h.batches.size() == 2; }, "latest pending frame not processed");
  check(h.batches[0]->capture.sequence == 1 && std::abs(h.batches[0]->spatial[0].position.z-1.01) < 1e-9 &&
        h.batches[1]->capture.sequence == 15 && std::abs(h.batches[1]->spatial[0].position.z-1.15) < 1e-9,
        "inference associated with the latest unrelated depth");
  h.until([&] { return h.status->replaced == 13 && h.status->processed == 2; }, "replacement accounting wrong");
  check(h.control->entered == 2, "obsolete pending frames accumulated");
}
void generation_during_inference() {
  Harness h;
  h.control->hold = true;
  const auto old = fixture(1,1000,"A"); h.send(old);
  h.until([&] { return h.control->entered == 1; }, "old inference did not begin");
  h.send(fixture(1,3000,"B")); h.control->hold = false;
  h.until([&] { return !h.batches.empty(); }, "new generation result missing");
  check(h.batches.size() == 1 && h.batches[0]->capture.generation == "B" &&
        h.batches[0]->calibration_id == "calibration_B" && h.batches[0]->spatial[0].position.z == 3,
        "old-generation result leaked or geometry crossed generations");
  h.send(old);
  h.until([&] { return h.status->superseded >= 1 && h.status->generation_changes == 1 && h.status->rejected >= 1; },
          "restart/late-frame accounting wrong");
  check(h.batches.size() == 1, "late old generation was reactivated");
}
void expiry_and_failure() {
  {
    Harness h(50);
    h.control->delay_ms = 120; h.send(fixture(1));
    h.until([&] { return h.status->stale >= 1; }, "slow result did not expire");
    check(h.batches.empty(), "expired result published");
    h.control->delay_ms = 0; h.send(fixture(2));
    h.until([&] { return h.batches.size() == 1; }, "fresh input did not recover after expiry");
    h.spin(300);
    check(h.status->state == "unavailable" && h.status->backend_ready, "input silence retained positive readiness");
  }
  {
    Harness h;
    h.control->fail = true; h.send(fixture(1));
    h.until([&] { return h.status->state == "error" && h.control->destroyed; }, "backend failure did not release resources");
    check(h.status->failures == 1 && !h.status->backend_ready && h.status->detail == "fixture_backend_failure", "failure not observable");
    h.send(fixture(2)); h.spin(30);
    check(h.control->entered == 1 && h.batches.empty(), "failed backend restarted implicitly");
  }
}
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    output_and_validation(); bounded_exact_association(); generation_during_inference(); expiry_and_failure();
    rclcpp::shutdown();
    std::cout << "detector adapter checks passed (synthetic backend, composed ROS)\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; rclcpp::shutdown(); return 1; }
}
