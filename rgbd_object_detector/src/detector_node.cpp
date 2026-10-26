// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#include "rgbd_object_detector/detector_node.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rgbd_camera_interfaces/msg/frame.hpp>
#include <rgbd_object_detector_interfaces/msg/batch.hpp>
#include <rgbd_object_detector_interfaces/msg/status.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <thread>

namespace rgbd_object_detector {
namespace {
using Frame = rgbd_camera_interfaces::msg::Frame;
using Batch = rgbd_object_detector_interfaces::msg::Batch;
using Status = rgbd_object_detector_interfaces::msg::Status;

template<class T> T setting(rclcpp::Node& node, const char* name, T value, const char* detail) {
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true; descriptor.description = detail;
  return node.declare_parameter<T>(name, value, descriptor);
}
int integer_setting(rclcpp::Node& node, const char* name, int value, const char* detail) {
  auto result = setting<int64_t>(node, name, value, detail);
  if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max())
    throw std::invalid_argument(std::string(name) + " outside integer range");
  return int(result);
}
void require(bool condition, const char* reason) {
  if (!condition) throw std::invalid_argument(reason);
}
ImageView image_view(const sensor_msgs::msg::Image& image) {
  return {image.data.data(), image.data.size(), image.width, image.height, image.step};
}
DepthView depth_view(const Frame& frame) {
  return {image_view(frame.depth), frame.depth_info.k, frame.depth_unit_m, frame.invalid_depth_value};
}
void validate_info(const sensor_msgs::msg::CameraInfo& info, const sensor_msgs::msg::Image& image) {
  require(info.header == image.header && info.width == image.width && info.height == image.height,
          "image/calibration header or size mismatch");
  require(info.distortion_model == "plumb_bob" && info.d.size() == 5 &&
          std::all_of(info.d.begin(), info.d.end(), [](double x) { return x == 0; }),
          "require rectified zero-distortion geometry");
  const std::array<double, 9> identity{1,0,0,0,1,0,0,0,1};
  const std::array<double, 12> projection{info.k[0],0,info.k[2],0,0,info.k[4],info.k[5],0,0,0,1,0};
  require(info.r == identity && info.p == projection, "unsupported rectification/projection");
  require(info.binning_x == 0 && info.binning_y == 0 && info.roi.x_offset == 0 && info.roi.y_offset == 0 &&
          info.roi.width == 0 && info.roi.height == 0 && !info.roi.do_rectify, "unsupported calibration ROI/binning");
}
void validate_stamp(const std_msgs::msg::Header& header, int64_t acquired, int64_t offset) {
  require(offset >= -acquired && offset <= std::numeric_limits<int64_t>::max() - acquired,
          "invalid acquisition-to-ROS mapping");
  require(header.stamp.sec >= 0 && header.stamp.nanosec < 1000000000 &&
          rclcpp::Time(header.stamp).nanoseconds() == acquired + offset, "acquisition/header timestamp mismatch");
}
void validate_frame(const Frame& frame, const std::string& clock, int64_t now) {
  const auto& capture = frame.capture;
  require(capture.schema_version == 1 && !capture.source_id.empty() && capture.source_id.size() <= 256 &&
          !capture.generation.empty() && capture.generation.size() <= 128 && capture.stream_id == "rgbd",
          "invalid capture identity");
  require(capture.clock_domain == clock, "unsupported_clock_domain");
  require(capture.acquisition_ns > 0 && frame.depth_acquisition_ns > 0 &&
          capture.acquisition_ns <= capture.received_ns && frame.depth_acquisition_ns <= capture.received_ns &&
          capture.received_ns <= now, "invalid or future acquisition/receipt time");
  require(frame.pairing_tolerance_ns > 0 && frame.pairing_tolerance_ns <= 50000000 &&
          frame.skew_ns == frame.depth_acquisition_ns - capture.acquisition_ns &&
          std::abs(frame.skew_ns) <= frame.pairing_tolerance_ns, "invalid_pairing");
  require(!frame.calibration_id.empty() && frame.calibration_id.size() <= 256, "missing calibration identity");
  require(frame.alignment == "aligned_to_rgb" && frame.depth_convention == "optical_z", "unsupported depth geometry");
  require(frame.rgb.encoding == "bgr8" && frame.depth.encoding == "16UC1" &&
          !frame.rgb.is_bigendian && !frame.depth.is_bigendian, "unsupported image encoding");
  image_view(frame.rgb).validate(3); depth_view(frame).validate();
  require(frame.rgb.width == frame.depth.width && frame.rgb.height == frame.depth.height &&
          frame.rgb_info.k == frame.depth_info.k, "RGB/depth virtual geometry mismatch");
  require(!frame.rgb.header.frame_id.empty() && frame.rgb.header.frame_id.size() <= 256 &&
          frame.rgb.header.frame_id == frame.depth.header.frame_id, "RGB/depth coordinate frame mismatch");
  validate_info(frame.rgb_info, frame.rgb); validate_info(frame.depth_info, frame.depth);
  validate_stamp(frame.rgb.header, capture.acquisition_ns, capture.unix_offset_ns);
  validate_stamp(frame.depth.header, frame.depth_acquisition_ns, capture.unix_offset_ns);
}

std::unique_ptr<Batch> make_batch(const Frame& frame, const Decoded& decoded,
                                const BackendIdentity& backend, const Config& config, const ModelProfile& profile) {
  auto result = std::make_unique<Batch>();
  result->capture = frame.capture;
  result->depth_sequence = frame.depth_sequence; result->depth_acquisition_ns = frame.depth_acquisition_ns;
  result->skew_ns = frame.skew_ns; result->pairing_tolerance_ns = frame.pairing_tolerance_ns;
  result->calibration_id = frame.calibration_id; result->camera_backend = frame.backend;
  result->model_id = backend.model_id; result->model_sha256 = backend.model_sha256;
  result->engine_sha256 = backend.engine_sha256; result->inference_backend = backend.name;
  result->model_profile_sha256 = profile.profile_sha256;
  result->preprocessing = profile.preprocessing();
  result->source_width = frame.rgb.width; result->source_height = frame.rgb.height;
  result->model_width = kModelWidth; result->model_height = kModelHeight;
  const auto transform = image_transform(profile, frame.rgb.width, frame.rgb.height);
  result->model_to_source_x = transform.scale_x; result->model_to_source_y = transform.scale_y;
  result->model_to_source_offset_x = transform.offset_x; result->model_to_source_offset_y = transform.offset_y;
  result->score_threshold = config.score_threshold; result->iou_threshold = config.iou_threshold;
  result->roi_fraction = config.roi_fraction; result->min_depth_m = config.min_depth_m;
  result->max_depth_m = config.max_depth_m; result->min_valid_pixels = config.min_valid_pixels;
  result->min_valid_fraction = config.min_valid_fraction;
  result->max_frame_age_ns = int64_t(config.max_frame_age_ms * 1e6);
  result->candidate_count = decoded.candidate_count; result->nms_candidate_count = decoded.nms_candidate_count;
  result->nms_count = decoded.nms_count; result->truncated = decoded.truncated;
  result->detections.header = frame.rgb.header;
  for (const auto& box : decoded.boxes) {
    vision_msgs::msg::Detection2D detection;
    detection.header = frame.rgb.header; // ID deliberately empty: no tracking identity.
    detection.bbox.center.position.x = (box.x1 + box.x2) / 2;
    detection.bbox.center.position.y = (box.y1 + box.y2) / 2;
    detection.bbox.size_x = box.x2 - box.x1; detection.bbox.size_y = box.y2 - box.y1;
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = profile.classes.at(box.class_index);
    hypothesis.hypothesis.score = box.score;
    hypothesis.pose.pose.position.x = hypothesis.pose.pose.position.y = hypothesis.pose.pose.position.z = NAN;
    hypothesis.pose.pose.orientation.x = hypothesis.pose.pose.orientation.y = NAN;
    hypothesis.pose.pose.orientation.z = hypothesis.pose.pose.orientation.w = NAN;
    hypothesis.pose.covariance.fill(NAN);
    detection.results.push_back(std::move(hypothesis));
    const auto spatial = estimate(box, depth_view(frame), config);
    rgbd_object_detector_interfaces::msg::Spatial output;
    output.detection_index = result->detections.detections.size();
    output.valid = spatial.valid; output.invalid_reason = spatial.reason; output.method = kSpatialMethod;
    output.position.x = spatial.xyz[0]; output.position.y = spatial.xyz[1]; output.position.z = spatial.xyz[2];
    output.pixel_u = spatial.u; output.pixel_v = spatial.v;
    output.roi_pixels = spatial.roi_pixels; output.valid_pixels = spatial.valid_pixels;
    output.valid_fraction = spatial.valid_fraction; output.depth_mad_m = spatial.depth_mad_m;
    result->detections.detections.push_back(std::move(detection)); result->spatial.push_back(std::move(output));
  }
  return result;
}
}

class DetectorNode::Implementation {
 public:
  Implementation(DetectorNode& node, BackendFactory factory) : node_(node), factory_(std::move(factory)) {
    if (node.get_parameter("use_sim_time").as_bool()) throw std::invalid_argument("physical detector cannot use simulated time");
    engine_path_ = setting<std::string>(node, "engine_path", "", "Target-built engine, with same-path .json manifest");
    if (engine_path_.empty()) throw std::invalid_argument("engine_path is required");
    const auto profile_path = setting<std::string>(node, "model_profile",
        ament_index_cpp::get_package_share_directory("rgbd_object_detector") + "/models/cones.json",
        "Validated model profile JSON; selection requires a controlled restart");
    profile_ = load_model_profile(profile_path);
    config_.score_threshold = setting<double>(node, "score_threshold", profile_.score_threshold, "Inclusive class score threshold, 0..1");
    config_.iou_threshold = setting<double>(node, "iou_threshold", profile_.iou_threshold, "Class-aware NMS IoU threshold, 0..1");
    config_.pre_nms_limit = integer_setting(node, "pre_nms_limit", profile_.pre_nms_limit, "Maximum scored candidates retained before NMS, 1..8400");
    config_.max_detections = integer_setting(node, "max_detections", profile_.max_detections, "Maximum output count, 1..100; truncation is explicit");
    config_.roi_fraction = setting<double>(node, "roi_fraction", 0.5, "Central width/height fraction of each box, 0.01..1");
    config_.min_depth_m = setting<double>(node, "min_depth_m", 0.1, "Minimum accepted optical Z, metres");
    config_.max_depth_m = setting<double>(node, "max_depth_m", 5.0, "Maximum accepted optical Z, metres");
    config_.min_valid_pixels = integer_setting(node, "min_valid_pixels", 5, "Minimum valid depth samples");
    config_.min_valid_fraction = setting<double>(node, "min_valid_fraction", 0.05, "Minimum valid fraction of sampled region");
    config_.max_frame_age_ms = setting<double>(node, "max_frame_age_ms", 500.0, "Maximum capture age at admission and completion, 10..2000 ms");
    config_.validate();
    status_.clock_domain = local_clock_domain(); status_.state = "starting";
    status_.model_id = profile_.model_id; status_.model_sha256 = profile_.onnx_sha256;
    status_.model_profile_sha256 = profile_.profile_sha256;
    status_.max_frame_age_ns = int64_t(config_.max_frame_age_ms * 1e6);
    const auto qos = rclcpp::SensorDataQoS().keep_last(1);
    batches_ = node.create_publisher<Batch>("~/batches", qos);
    status_publisher_ = node.create_publisher<Status>("~/status", rclcpp::QoS(1).best_effort());
    subscription_ = node.create_subscription<Frame>("frames", qos, [this](Frame::ConstSharedPtr frame) { receive(std::move(frame)); });
    timer_ = node.create_wall_timer(std::chrono::milliseconds(200), [this] { publish_status(); });
    worker_ = std::thread([this] { run(); });
  }
  ~Implementation() {
    timer_->cancel(); subscription_.reset();
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; pending_.reset(); }
    available_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

 private:
  void reject(const std::string& reason) { ++status_.rejected; status_.detail = reason; }
  void receive(Frame::ConstSharedPtr frame) {
    const auto now = monotonic_ns();
    std::string error;
    try { validate_frame(*frame, status_.clock_domain, now); }
    catch (const std::exception& e) { error = e.what(); }
    std::lock_guard<std::mutex> lock(mutex_);
    ++status_.received; status_.last_received_ns = now;
    if (stopping_ || status_.state == "error") return;
    if (!error.empty()) { reject(error); return; }
    if (now - frame->capture.acquisition_ns > status_.max_frame_age_ns) { ++status_.stale; status_.detail = "stale_input"; return; }
    if (!status_.source_id.empty() && frame->capture.source_id != status_.source_id) { reject("source_changed_restart_required"); return; }
    if (frame->capture.acquisition_ns <= status_.last_input_acquisition_ns) { reject("duplicate_or_regressed_acquisition"); return; }
    if (frame->capture.generation == status_.generation) {
      if (frame->capture.sequence <= last_sequence_) { reject("duplicate_or_regressed_sequence"); return; }
      if (frame->calibration_id != status_.calibration_id) { reject("calibration_changed_without_generation"); return; }
    } else {
      if (!status_.generation.empty()) ++status_.generation_changes;
      if (pending_) { pending_.reset(); ++status_.superseded; }
      ++epoch_;
      status_.generation = frame->capture.generation; status_.source_id = frame->capture.source_id;
      status_.calibration_id = frame->calibration_id; status_.last_processed_acquisition_ns = 0;
      status_.last_completed_ns = 0;
      status_.last_preprocess_ns = status_.last_inference_ns = status_.last_postprocess_ns = status_.last_source_age_ns = 0;
    }
    last_sequence_ = frame->capture.sequence; status_.last_input_acquisition_ns = frame->capture.acquisition_ns;
    if (pending_) ++status_.replaced;
    pending_ = std::move(frame); status_.pending = 1;
    available_.notify_one();
  }
  void publish_status() {
    Status output;
    { std::lock_guard<std::mutex> lock(mutex_); output = status_; }
    output.header.stamp = node_.now(); output.observed_ns = monotonic_ns(); output.expires_after_ns = 1000000000;
    if (output.backend_ready && output.state != "error") {
      if (output.last_processed_acquisition_ns > 0 &&
          output.observed_ns - output.last_processed_acquisition_ns <= output.max_frame_age_ns) output.state = "streaming";
      else {
        output.state = "unavailable";
        if (output.detail.empty()) output.detail = "no_recent_successfully_processed_capture";
      }
    }
    status_publisher_->publish(output);
  }
  void run() {
    std::unique_ptr<InferenceBackend> backend;
    try {
      backend = factory_(engine_path_, profile_);
      if (!backend || !backend->input()) throw std::runtime_error("backend returned no input storage");
      const auto identity = backend->identity();
      if (identity.model_id != profile_.model_id || identity.model_sha256 != profile_.onnx_sha256 ||
          identity.profile_sha256 != profile_.profile_sha256 || identity.engine_sha256.empty())
        throw std::runtime_error("backend model/engine identity mismatch");
      {
        std::lock_guard<std::mutex> lock(mutex_);
        status_.backend_ready = true; status_.state = "unavailable";
        status_.inference_backend = identity.name; status_.engine_sha256 = identity.engine_sha256;
      }
      for (;;) {
        Frame::ConstSharedPtr frame;
        uint64_t epoch;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          available_.wait(lock, [this] { return stopping_ || bool(pending_); });
          if (stopping_) break;
          frame = std::move(pending_); epoch = epoch_;
          status_.pending = 0; status_.processing = true;
        }
        const auto started = monotonic_ns();
        if (started - frame->capture.acquisition_ns > int64_t(config_.max_frame_age_ms * 1e6)) {
          std::lock_guard<std::mutex> lock(mutex_); ++status_.stale; status_.processing = false; continue;
        }
        preprocess(image_view(frame->rgb), backend->input(), kInputElements, profile_);
        const auto preprocessed = monotonic_ns();
        const auto* raw = backend->infer();
        const auto inferred = monotonic_ns();
        const auto decoded = decode(raw, profile_.output_elements(), frame->rgb.width, frame->rgb.height, config_, profile_);
        auto output = make_batch(*frame, decoded, identity, config_, profile_);
        output->started_ns = started; output->completed_ns = monotonic_ns();
        output->preprocess_ns = preprocessed - started; output->inference_ns = inferred - preprocessed;
        output->postprocess_ns = output->completed_ns - inferred;
        output->source_age_ns = output->completed_ns - frame->capture.acquisition_ns;
        std::lock_guard<std::mutex> lock(mutex_);
        status_.processing = false;
        if (stopping_) break;
        if (epoch != epoch_) { ++status_.superseded; continue; }
        if (monotonic_ns() - frame->capture.acquisition_ns > status_.max_frame_age_ns) {
          ++status_.stale; status_.detail = "stale_result"; continue;
        }
        status_.last_preprocess_ns = output->preprocess_ns; status_.last_inference_ns = output->inference_ns;
        status_.last_postprocess_ns = output->postprocess_ns; status_.last_source_age_ns = output->source_age_ns;
        status_.last_completed_ns = output->completed_ns;
        // Linearize publication with generation changes. This is a bounded small
        // message; GPU work and image operations never run under this mutex.
        batches_->publish(std::move(output));
        ++status_.processed; status_.last_processed_acquisition_ns = frame->capture.acquisition_ns; status_.detail.clear();
      }
    } catch (const std::exception& error) {
      std::lock_guard<std::mutex> lock(mutex_);
      status_.state = "error"; status_.detail = error.what(); ++status_.failures;
      status_.backend_ready = false; status_.processing = false; status_.pending = 0; pending_.reset();
      RCLCPP_ERROR(node_.get_logger(), "%s", error.what());
    }
    backend.reset(); // On the owning worker, including terminal faults.
  }

  DetectorNode& node_;
  BackendFactory factory_;
  Config config_;
  ModelProfile profile_;
  std::string engine_path_;
  std::mutex mutex_;
  std::condition_variable available_;
  bool stopping_ = false;
  uint64_t epoch_ = 0, last_sequence_ = 0;
  Frame::ConstSharedPtr pending_;
  Status status_;
  std::thread worker_;
  rclcpp::Subscription<Frame>::SharedPtr subscription_;
  rclcpp::Publisher<Batch>::SharedPtr batches_;
  rclcpp::Publisher<Status>::SharedPtr status_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

DetectorNode::DetectorNode(const rclcpp::NodeOptions& options) : DetectorNode(options, make_tensorrt_backend) {}
DetectorNode::DetectorNode(const rclcpp::NodeOptions& options, BackendFactory factory)
    : Node("rgbd_object_detector", options), impl_(std::make_unique<Implementation>(*this, std::move(factory))) {}
DetectorNode::~DetectorNode() = default;
}  // namespace rgbd_object_detector
RCLCPP_COMPONENTS_REGISTER_NODE(rgbd_object_detector::DetectorNode)
