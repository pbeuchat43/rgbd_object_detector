// Manual native engine acceptance, requiring an explicitly supplied local engine.
#include "rgbd_object_detector/core.hpp"
#include "rgbd_object_detector/tensorrt_support.hpp"
#include <algorithm>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
  using namespace rgbd_object_detector;
  try {
    if (argc != 2) throw std::runtime_error("usage: test_engine /absolute/cones.engine");
    // Exercise shared logger lifetime across builder and independent runtimes.
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(shared_tensorrt_logger()));
    if (!builder) throw std::runtime_error("builder creation failed");
    auto first = make_tensorrt_backend(argv[1]);
    auto second = make_tensorrt_backend(argv[1]);
    first.reset(); builder.reset();
    std::fill_n(second->input(), kInputElements, 0.0F);
    const auto result = decode(second->infer(), kOutputElements, 640, 640, Config{});
    std::cout << "Engine checked: " << second->identity().engine_sha256 << ", zero-image candidates " << result.candidate_count << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
