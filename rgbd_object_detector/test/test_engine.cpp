// Manual native engine acceptance, requiring an explicitly supplied local engine.
#include "rgbd_object_detector/core.hpp"
#include "rgbd_object_detector/tensorrt_support.hpp"
#include <algorithm>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
  using namespace rgbd_object_detector;
  try {
    if (argc != 3) throw std::runtime_error("usage: test_engine /absolute/model.engine /absolute/model-profile.json");
    const auto profile = load_model_profile(argv[2]);
    // Exercise shared logger lifetime across builder and independent runtimes.
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(shared_tensorrt_logger()));
    if (!builder) throw std::runtime_error("builder creation failed");
    auto first = make_tensorrt_backend(argv[1], profile);
    auto second = make_tensorrt_backend(argv[1], profile);
    first.reset(); builder.reset();
    std::fill_n(second->input(), kInputElements, 0.0F);
    const auto result = decode(second->infer(), profile.output_elements(), 640, 640, Config{}, profile);
    std::cout << "Engine checked: " << second->identity().engine_sha256 << ", zero-image candidates " << result.candidate_count << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
