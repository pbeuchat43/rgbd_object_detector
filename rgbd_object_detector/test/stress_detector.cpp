// Manual hardware-test driver, never installed or registered as a component.
#include "rgbd_object_detector/detector_node.hpp"
#include <chrono>
#include <thread>

using namespace rgbd_object_detector;
class DelayedBackend : public InferenceBackend {
 public:
  DelayedBackend(const std::string& path, const ModelProfile& profile) : backend_(make_tensorrt_backend(path, profile)) {}
  float* input() override { return backend_->input(); }
  const float* infer() override {
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    return backend_->infer();
  }
  BackendIdentity identity() const override {
    auto result = backend_->identity(); result.name += ":test_delay_350ms"; return result;
  }
 private:
  std::unique_ptr<InferenceBackend> backend_;
};
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DetectorNode>(rclcpp::NodeOptions(),
      [](const std::string& path, const ModelProfile& profile) { return std::make_unique<DelayedBackend>(path, profile); });
  rclcpp::spin(node);
  node.reset(); rclcpp::shutdown();
  return 0;
}
