// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#include "rgbd_object_detector/overlay.hpp"
#include "rgbd_object_detector/core.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace rgbd_object_detector {
namespace {
void label(cv::Mat& image, const std::string& text, int x, int y, const cv::Scalar& color) {
  int baseline = 0;
  const auto size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, .45, 1, &baseline);
  x = std::clamp(x, 0, std::max(0, image.cols - size.width - 4));
  y = std::clamp(y, size.height + 2, std::max(size.height + 2, image.rows - baseline - 2));
  cv::rectangle(image, cv::Rect(x, y - size.height - 2, std::min(size.width + 4, image.cols - x),
                             std::min(size.height + baseline + 4, image.rows - y + size.height + 2)), cv::Scalar(0,0,0), cv::FILLED);
  cv::putText(image, text, cv::Point(x + 2, y), cv::FONT_HERSHEY_SIMPLEX, .45, color, 1, cv::LINE_AA);
}
// Optional optical measurement marker: 0xA5, acquisition ROS ns (little endian),
// capture sequence low 32 bits, then CRC-16/CCITT over the first 13 bytes.
// 120 black/white cells, two rows of 60 at (8, height-24), each 8x8 pixels.
// This binds a decoded browser frame to its original acquisition, without
// pretending WebRTC's transport timestamps already preserve camera identity.
void marker(cv::Mat& image, const rgbd_object_detector_interfaces::msg::Batch& batch) {
  if (image.cols < 496 || image.rows < 32) throw std::invalid_argument("capture marker requires at least 496x32 pixels");
  std::array<uint8_t, 15> bytes{};
  bytes[0] = 0xA5;
  const auto& stamp = batch.detections.header.stamp;
  const uint64_t ns = uint64_t(stamp.sec) * 1000000000ULL + stamp.nanosec;
  for (int i = 0; i < 8; ++i) bytes[1+i] = (ns >> (i*8)) & 255;
  for (int i = 0; i < 4; ++i) bytes[9+i] = (batch.capture.sequence >> (i*8)) & 255;
  uint16_t crc = 0xffff;
  for (int i = 0; i < 13; ++i) {
    crc ^= uint16_t(bytes[i]) << 8;
    for (int bit = 0; bit < 8; ++bit) crc = (crc & 0x8000) ? uint16_t((crc << 1) ^ 0x1021) : uint16_t(crc << 1);
  }
  bytes[13] = crc & 255; bytes[14] = crc >> 8;
  for (int bit = 0; bit < 120; ++bit) {
    const int value = (bytes[bit/8] & (1 << (bit%8))) ? 255 : 0;
    cv::rectangle(image, cv::Rect(8 + (bit%60)*8, image.rows - 24 + (bit/60)*8, 8, 8), cv::Scalar::all(value), cv::FILLED);
  }
}
void draw_batch(cv::Mat& image, const sensor_msgs::msg::Image& source,
    const rgbd_object_detector_interfaces::msg::Batch& batch, bool spatial, const cv::Scalar* model_color = nullptr) {
  ImageView{source.data.data(), source.data.size(), source.width, source.height, source.step}.validate(3);
  if (source.encoding != "bgr8" || source.is_bigendian || source.header != batch.detections.header ||
      source.width != batch.source_width || source.height != batch.source_height ||
      batch.detections.detections.size() > 100 || batch.spatial.size() != batch.detections.detections.size())
    throw std::invalid_argument("overlay source/result contract mismatch");
  for (size_t i = 0; i < batch.detections.detections.size(); ++i) {
    const auto& detection = batch.detections.detections[i];
    const auto& box = detection.bbox;
    if (detection.header != source.header || detection.results.size() != 1 || batch.spatial[i].detection_index != i)
      throw std::invalid_argument("overlay detection association mismatch");
    const double x1 = box.center.position.x - box.size_x / 2, y1 = box.center.position.y - box.size_y / 2;
    const double x2 = box.center.position.x + box.size_x / 2, y2 = box.center.position.y + box.size_y / 2;
    const auto& hypothesis = detection.results[0].hypothesis;
    if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2) ||
        x1 < 0 || y1 < 0 || x2 > source.width || y2 > source.height || x2 <= x1 || y2 <= y1 ||
        !std::isfinite(hypothesis.score) || hypothesis.score < 0 || hypothesis.score > 1 || hypothesis.class_id.size() > 80)
      throw std::invalid_argument("invalid overlay detection");
    const cv::Scalar color = model_color ? *model_color :
        (batch.spatial[i].valid ? cv::Scalar(80,230,80) : cv::Scalar(0,210,255));
    cv::rectangle(image, cv::Point(int(std::floor(x1)), int(std::floor(y1))),
                  cv::Point(std::min(int(source.width)-1, int(std::ceil(x2))-1),
                            std::min(int(source.height)-1, int(std::ceil(y2))-1)), color, 2);
    std::ostringstream text;
    text << hypothesis.class_id << ' ' << std::fixed << std::setprecision(2) << hypothesis.score;
    if (spatial) {
      if (batch.spatial[i].valid && std::isfinite(batch.spatial[i].position.z))
        text << "  Z " << batch.spatial[i].position.z << "m";
      else text << "  2D only";
    }
    label(image, text.str(), int(x1), int(y1) - 4, color);
  }
}
}

std::unique_ptr<sensor_msgs::msg::Image> render_overlay(const sensor_msgs::msg::Image& source,
    const rgbd_object_detector_interfaces::msg::Batch& batch, bool spatial, bool capture_marker) {
  ImageView{source.data.data(), source.data.size(), source.width, source.height, source.step}.validate(3);
  auto output = std::make_unique<sensor_msgs::msg::Image>(source);
  cv::Mat image(source.height, source.width, CV_8UC3, output->data.data(), source.step);
  draw_batch(image, source, batch, spatial);
  label(image, batch.model_id.substr(0, 65), 2, 16, cv::Scalar(255,255,255));
  std::ostringstream state;
  state << "capture " << batch.capture.sequence << "  " << batch.detections.detections.size() << " detections";
  if (batch.truncated) state << "  TRUNCATED (NMS " << batch.nms_count << ')';
  label(image, state.str(), 2, 34, batch.truncated ? cv::Scalar(0,80,255) : cv::Scalar(255,255,255));
  if (capture_marker) marker(image, batch);
  return output;
}

std::unique_ptr<sensor_msgs::msg::Image> render_combined_overlay(const sensor_msgs::msg::Image& source,
    const rgbd_object_detector_interfaces::msg::Batch& first,
    const rgbd_object_detector_interfaces::msg::Batch& second, bool spatial, bool capture_marker) {
  if (first.capture != second.capture || first.calibration_id != second.calibration_id ||
      first.depth_sequence != second.depth_sequence || first.depth_acquisition_ns != second.depth_acquisition_ns ||
      first.skew_ns != second.skew_ns || first.pairing_tolerance_ns != second.pairing_tolerance_ns ||
      first.camera_backend != second.camera_backend)
    throw std::invalid_argument("combined overlay requires identical capture and pairing metadata");
  ImageView{source.data.data(), source.data.size(), source.width, source.height, source.step}.validate(3);
  auto output = std::make_unique<sensor_msgs::msg::Image>(source);
  cv::Mat image(source.height, source.width, CV_8UC3, output->data.data(), source.step);
  const std::array<cv::Scalar, 2> colors{cv::Scalar(0,210,255), cv::Scalar(255,210,80)};
  const std::array<const rgbd_object_detector_interfaces::msg::Batch*, 2> batches{&first, &second};
  for (size_t i = 0; i < batches.size(); ++i) draw_batch(image, source, *batches[i], spatial, &colors[i]);
  for (size_t i = 0; i < batches.size(); ++i) {
    const auto& batch = *batches[i];
    std::ostringstream text;
    text << batch.model_id.substr(0, 55) << "  " << batch.detections.detections.size() << " detections";
    if (batch.truncated) text << "  TRUNCATED (NMS " << batch.nms_count << ')';
    label(image, text.str(), 2, 16 + int(i)*18, colors[i]);
  }
  label(image, "capture " + std::to_string(first.capture.sequence) + "  both models", 2, 52, cv::Scalar(255,255,255));
  if (capture_marker) marker(image, first);
  return output;
}

OverlayWorker::OverlayWorker(rclcpp::Node& node, bool spatial, bool capture_marker, Render renderer)
    : spatial_(spatial), capture_marker_(capture_marker), renderer_(std::move(renderer)) {
  const auto qos = rclcpp::SensorDataQoS().keep_last(1);
  images_ = node.create_publisher<sensor_msgs::msg::Image>("~/overlay/image", qos);
  frames_ = node.create_publisher<rgbd_object_detector_interfaces::msg::OverlayFrame>("~/overlay/frame", qos);
  worker_ = std::thread([this] { run(); });
}
OverlayWorker::~OverlayWorker() {
  stopping_ = true;
  { std::lock_guard<std::mutex> lock(mutex_); pending_.reset(); }
  available_.notify_all();
  if (worker_.joinable()) worker_.join();
}
bool OverlayWorker::has_consumers() const {
  return images_->get_subscription_count() + images_->get_intra_process_subscription_count() +
      frames_->get_subscription_count() + frames_->get_intra_process_subscription_count() > 0;
}
void OverlayWorker::invalidate(uint64_t epoch) {
  epoch_.store(epoch);
  std::lock_guard<std::mutex> lock(mutex_);
  if (pending_) { pending_.reset(); ++statistics_.superseded; }
  statistics_.pending = 0; statistics_.last_acquisition_ns = 0;
}
void OverlayWorker::submit(Frame::ConstSharedPtr frame, Batch::ConstSharedPtr batch, uint64_t epoch) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_ || !statistics_.error.empty()) return;
  if (!has_consumers()) { ++statistics_.no_consumers; return; }
  if (epoch != epoch_.load()) { ++statistics_.superseded; return; }
  if (pending_) ++statistics_.replaced;
  pending_ = std::make_unique<Job>(Job{std::move(frame), std::move(batch), epoch});
  statistics_.pending = 1; available_.notify_one();
}
OverlayStatistics OverlayWorker::statistics() {
  std::lock_guard<std::mutex> lock(mutex_); return statistics_;
}
void OverlayWorker::run() {
  try {
    for (;;) {
      std::unique_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        available_.wait(lock, [this] { return stopping_ || bool(pending_); });
        if (stopping_) return;
        job = std::move(pending_); statistics_.pending = 0; statistics_.processing = true;
      }
      const auto started = monotonic_ns();
      std::unique_ptr<sensor_msgs::msg::Image> image;
      const auto& batch = *job->batch;
      if (job->epoch == epoch_.load() && started - batch.capture.acquisition_ns <= batch.max_frame_age_ns && has_consumers()) {
        if (job->frame->capture != batch.capture || job->frame->calibration_id != batch.calibration_id)
          throw std::invalid_argument("overlay requires the exact retained inference frame");
        image = renderer_(job->frame->rgb, batch, spatial_, capture_marker_);
      }
      const auto completed = monotonic_ns();
      const bool superseded = job->epoch != epoch_.load();
      const bool stale = completed - batch.capture.acquisition_ns > batch.max_frame_age_ns;
      // Publication and rendering never hold the queue mutex. A slow external
      // subscriber cannot make inference wait for rendering or middleware work.
      if (image && !superseded && !stale && !stopping_) {
        if (frames_->get_subscription_count() + frames_->get_intra_process_subscription_count() > 0) {
          auto atomic = std::make_unique<rgbd_object_detector_interfaces::msg::OverlayFrame>();
          atomic->batch = batch; atomic->image = *image;
          atomic->render_started_ns = started; atomic->render_completed_ns = completed;
          frames_->publish(std::move(atomic));
        }
        images_->publish(std::move(image));
        std::lock_guard<std::mutex> lock(mutex_);
        ++statistics_.rendered; statistics_.last_render_ns = completed - started;
        if (job->epoch == epoch_.load()) statistics_.last_acquisition_ns = batch.capture.acquisition_ns;
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        statistics_.processing = false;
        if (superseded) ++statistics_.superseded;
        else if (stale) ++statistics_.stale;
      }
    }
  } catch (const std::exception& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    statistics_.error = error.what(); ++statistics_.failures;
    statistics_.processing = false; statistics_.pending = 0; pending_.reset();
  }
}
}  // namespace rgbd_object_detector
