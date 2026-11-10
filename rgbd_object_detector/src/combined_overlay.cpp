// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#include "rgbd_object_detector/combined_overlay.hpp"
#include "rgbd_object_detector/core.hpp"
#include <rclcpp_components/register_node_macro.hpp>
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace rgbd_object_detector {
CombinedOverlayNode::CombinedOverlayNode(const rclcpp::NodeOptions& options, Render renderer)
    : Node("combined_overlay", options), renderer_(std::move(renderer)) {
  rcl_interfaces::msg::ParameterDescriptor read_only;
  read_only.read_only = true;
  const auto capacity = declare_parameter<int>("join_capacity", 8, read_only);
  const auto age = declare_parameter<int>("max_frame_age_ms", 500, read_only);
  if (capacity < 2 || capacity > 32 || age < 10 || age > 2000 || get_parameter("use_sim_time").as_bool())
    throw std::invalid_argument("combined overlay requires capacity 2..32, age 10..2000 ms and physical time");
  capacity_ = size_t(capacity); max_age_ns_ = int64_t(age)*1000000;
  spatial_ = declare_parameter<bool>("overlay_spatial", true, read_only);
  marker_ = declare_parameter<bool>("overlay_capture_marker", false, read_only);
  status_.clock_domain = local_clock_domain(); status_.capacity = capacity_;
  const auto qos = rclcpp::SensorDataQoS().keep_last(1);
  images_ = create_publisher<sensor_msgs::msg::Image>("~/overlay/image", qos);
  frames_ = create_publisher<rgbd_object_detector_interfaces::msg::CombinedOverlayFrame>("~/overlay/frame", qos);
  statuses_ = create_publisher<Status>("~/status", qos);
  input_ = create_subscription<Frame>("frames", qos, [this](Frame::ConstSharedPtr frame) { receive_frame(std::move(frame)); });
  for (size_t i = 0; i < 2; ++i)
    capture_detections_[i] = create_subscription<CaptureDetections>(i == 0 ? "first_capture_detections" : "second_capture_detections", qos,
        [this, i](CaptureDetections::ConstSharedPtr capture_detections) { receive_capture_detections(i, std::move(capture_detections)); });
  timer_ = create_wall_timer(std::chrono::milliseconds(200), [this] { publish_status(); });
  worker_ = std::thread([this] { run(); });
}
CombinedOverlayNode::~CombinedOverlayNode() {
  stopping_ = true;
  { std::lock_guard<std::mutex> lock(mutex_); clear(); }
  available_.notify_all();
  if (worker_.joinable()) worker_.join();
}
bool CombinedOverlayNode::has_consumers() const {
  return images_->get_subscription_count() + images_->get_intra_process_subscription_count() +
      frames_->get_subscription_count() + frames_->get_intra_process_subscription_count() > 0;
}
bool CombinedOverlayNode::capture_valid(const rgbd_camera_interfaces::msg::CaptureReference& c, int64_t now) const {
  return c.schema_version == 1 && !c.source_id.empty() && c.source_id.size() <= 128 &&
      !c.generation.empty() && c.generation.size() <= 128 && c.stream_id == "rgbd" &&
      c.clock_domain == status_.clock_domain && c.acquisition_ns > 0 && c.acquisition_ns <= now &&
      now - c.acquisition_ns <= max_age_ns_ && c.received_ns >= c.acquisition_ns && c.received_ns <= now;
}
bool CombinedOverlayNode::fresh(const Entry& entry, int64_t now) const {
  auto limit = max_age_ns_;
  for (const auto& capture_detections : entry.capture_detections) if (capture_detections) limit = std::min(limit, capture_detections->max_frame_age_ns);
  return entry.acquisition_ns > 0 && entry.acquisition_ns <= now && now - entry.acquisition_ns <= limit;
}
void CombinedOverlayNode::clear() {
  ++epoch_;
  status_.superseded += entries_.size() + (pending_ ? 1 : 0);
  entries_.clear(); pending_.reset(); status_.last_acquisition_ns = 0; last_output_deadline_ = 0;
}
void CombinedOverlayNode::expire(int64_t now) {
  for (auto it = entries_.begin(); it != entries_.end();) {
    if (!fresh(it->second, now)) { it = entries_.erase(it); ++status_.expired; }
    else ++it;
  }
}
void CombinedOverlayNode::receive_frame(Frame::ConstSharedPtr frame) {
  const auto now = monotonic_ns();
  std::lock_guard<std::mutex> lock(mutex_); ++status_.received[0];
  const auto& c = frame->capture;
  try {
    ImageView{frame->rgb.data.data(), frame->rgb.data.size(), frame->rgb.width, frame->rgb.height, frame->rgb.step}.validate(3);
    if (!capture_valid(c, now) || frame->rgb.encoding != "bgr8" || frame->rgb.is_bigendian ||
        frame->calibration_id.empty() || frame->calibration_id.size() > 256 ||
        (have_frame_ && (c.source_id != status_.source_id || c.acquisition_ns <= last_frame_acquisition_ ||
          (c.generation == status_.generation && (c.sequence <= last_frame_sequence_ || frame->calibration_id != calibration_)))))
      throw std::invalid_argument("invalid or regressed combined source");
  } catch (const std::exception&) { ++status_.rejected; return; }
  if (!have_frame_ || c.generation != status_.generation) {
    clear(); have_capture_detections_ = {}; have_match_ = false;
    status_.source_id = c.source_id; status_.generation = c.generation; calibration_ = frame->calibration_id;
  }
  have_frame_ = true; last_frame_sequence_ = c.sequence; last_frame_acquisition_ = c.acquisition_ns;
  if (!has_consumers()) { ++status_.no_consumers; clear(); return; }
  expire(now);
  if (have_match_ && c.sequence <= last_matched_sequence_) { ++status_.superseded; return; }
  auto& entry = entries_[c.sequence]; entry.frame = std::move(frame); entry.acquisition_ns = c.acquisition_ns;
  join(c.sequence);
}
void CombinedOverlayNode::receive_capture_detections(size_t index, CaptureDetections::ConstSharedPtr capture_detections) {
  const auto now = monotonic_ns();
  std::lock_guard<std::mutex> lock(mutex_); ++status_.received[index+1];
  const auto& c = capture_detections->capture;
  if (!have_frame_ || !capture_valid(c, now) || c.source_id != status_.source_id || c.generation != status_.generation ||
      capture_detections->schema_version != 1 || capture_detections->calibration_id != calibration_ || capture_detections->max_frame_age_ns <= 0 ||
      capture_detections->max_frame_age_ns > 2000000000LL || now - c.acquisition_ns > capture_detections->max_frame_age_ns ||
      capture_detections->model_id.empty() || capture_detections->model_id.size() > 128 || capture_detections->model_sha256.size() > 64 ||
      capture_detections->model_profile_sha256.size() > 64 || capture_detections->engine_sha256.size() > 64 ||
      capture_detections->detections.detections.size() > 100 || capture_detections->spatial.size() != capture_detections->detections.detections.size() ||
      (have_capture_detections_[index] && c.sequence <= last_capture_detections_sequence_[index])) { ++status_.rejected; return; }
  const auto identity = capture_detections->model_id + ':' + capture_detections->model_sha256 + ':' + capture_detections->model_profile_sha256 + ':' + capture_detections->engine_sha256;
  if (!model_identity_[index].empty() && model_identity_[index] != identity) { ++status_.rejected; return; }
  model_identity_[index] = identity; status_.model_ids[index] = capture_detections->model_id;
  have_capture_detections_[index] = true; last_capture_detections_sequence_[index] = c.sequence;
  if (!has_consumers()) { ++status_.no_consumers; clear(); return; }
  expire(now);
  if (have_match_ && c.sequence <= last_matched_sequence_) { ++status_.superseded; return; }
  auto& entry = entries_[c.sequence]; entry.capture_detections[index] = std::move(capture_detections); entry.acquisition_ns = c.acquisition_ns;
  join(c.sequence);
}
void CombinedOverlayNode::join(uint64_t sequence) {
  auto it = entries_.find(sequence);
  auto& e = it->second;
  if (e.frame && e.capture_detections[0] && e.capture_detections[1]) {
    const auto& f = *e.frame;
    for (const auto& capture_detections : e.capture_detections) {
      if (capture_detections->capture != f.capture || capture_detections->calibration_id != f.calibration_id || capture_detections->camera_backend != f.backend ||
          capture_detections->depth_sequence != f.depth_sequence || capture_detections->depth_acquisition_ns != f.depth_acquisition_ns ||
          capture_detections->skew_ns != f.skew_ns || capture_detections->pairing_tolerance_ns != f.pairing_tolerance_ns ||
          capture_detections->detections.header != f.rgb.header || capture_detections->source_width != f.rgb.width || capture_detections->source_height != f.rgb.height) {
        entries_.erase(it); ++status_.rejected; return;
      }
    }
    if (pending_) ++status_.replaced;
    pending_ = std::make_unique<Job>(Job{std::move(e), epoch_.load()});
    entries_.erase(it); ++status_.matched; have_match_ = true; last_matched_sequence_ = sequence;
    for (auto old = entries_.begin(); old != entries_.end() && old->first <= sequence;) {
      old = entries_.erase(old); ++status_.superseded;
    }
    available_.notify_one();
  }
  while (entries_.size() > capacity_) { entries_.erase(entries_.begin()); ++status_.evicted; }
}
CombinedOverlayNode::Status CombinedOverlayNode::statistics() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto result = status_; result.retained = entries_.size(); result.pending = pending_ ? 1 : 0;
  result.observed_ns = monotonic_ns(); result.expires_ns = result.observed_ns + 1000000000;
  result.state = !has_consumers() ? "idle" : !result.detail.empty() ? "error" :
      (result.last_acquisition_ns > 0 && result.observed_ns <= last_output_deadline_ ? "streaming" : "waiting_for_pair");
  return result;
}
void CombinedOverlayNode::publish_status() {
  { std::lock_guard<std::mutex> lock(mutex_); expire(monotonic_ns());
    if (!has_consumers() && (!entries_.empty() || pending_ || status_.processing)) clear(); }
  statuses_->publish(statistics());
}
void CombinedOverlayNode::run() {
  while (!stopping_) {
    std::unique_ptr<Job> job;
    { std::unique_lock<std::mutex> lock(mutex_);
      available_.wait(lock, [this] { return stopping_ || bool(pending_); });
      if (stopping_) return;
      job = std::move(pending_); status_.processing = true; }
    const auto started = monotonic_ns();
    bool rendered = false;
    int64_t completed = started;
    std::string error;
    try {
      if (job->epoch == epoch_.load() && fresh(job->entry, started) && has_consumers()) {
        const auto& e = job->entry;
        auto image = renderer_(e.frame->rgb, *e.capture_detections[0], *e.capture_detections[1], spatial_, marker_);
        completed = monotonic_ns();
        // Never hold the join mutex during drawing or middleware publication.
        if (job->epoch == epoch_.load() && fresh(e, completed) && has_consumers() && !stopping_) {
          const bool atomic_demand = frames_->get_subscription_count() + frames_->get_intra_process_subscription_count() > 0;
          const bool image_demand = images_->get_subscription_count() + images_->get_intra_process_subscription_count() > 0;
          if (atomic_demand) {
            auto atomic = std::make_unique<rgbd_object_detector_interfaces::msg::CombinedOverlayFrame>();
            atomic->capture_detections = {*e.capture_detections[0], *e.capture_detections[1]};
            if (image_demand) atomic->image = *image;
            else atomic->image = std::move(*image);
            atomic->render_started_ns = started; atomic->render_completed_ns = completed;
            frames_->publish(std::move(atomic));
          }
          if (image_demand) images_->publish(std::move(image));
          rendered = true;
        }
      }
    } catch (const std::exception& failure) { error = failure.what(); }
    { std::lock_guard<std::mutex> lock(mutex_);
      status_.processing = false;
      if (!error.empty()) { ++status_.failures; status_.detail = error.substr(0, 256); }
      else if (job->epoch != epoch_.load()) ++status_.superseded;
      else if (!fresh(job->entry, monotonic_ns())) ++status_.expired;
      else if (rendered) {
        ++status_.rendered; status_.last_acquisition_ns = job->entry.acquisition_ns;
        last_output_deadline_ = job->entry.acquisition_ns + std::min({max_age_ns_,
            job->entry.capture_detections[0]->max_frame_age_ns, job->entry.capture_detections[1]->max_frame_age_ns});
        status_.last_render_ns = completed - started; status_.detail.clear();
      }
    }
  }
}
}  // namespace rgbd_object_detector
RCLCPP_COMPONENTS_REGISTER_NODE(rgbd_object_detector::CombinedOverlayNode)
