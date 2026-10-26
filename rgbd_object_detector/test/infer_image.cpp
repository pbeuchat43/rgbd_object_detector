// Saved-image reference harness. Opens the explicitly supplied engine, never a camera.
// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#include "rgbd_object_detector/core.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  using namespace rgbd_object_detector;
  try {
    if (argc != 5) throw std::runtime_error("usage: infer_image engine profile.json image.png output-prefix");
    const auto profile = load_model_profile(argv[2]);
    const cv::Mat image = cv::imread(argv[3], cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("cannot read image");
    const std::string prefix = argv[4];
    for (const auto* suffix : {".input.f32", ".output.f32", ".json"})
      if (std::filesystem::exists(prefix + suffix)) throw std::runtime_error("refusing to overwrite reference output");
    auto backend = make_tensorrt_backend(argv[1], profile);
    preprocess({image.data, image.step * size_t(image.rows), uint32_t(image.cols), uint32_t(image.rows), uint32_t(image.step)},
               backend->input(), kInputElements, profile);
    auto write = [&](const char* suffix, const float* data, size_t elements) {
      std::ofstream file(prefix + suffix, std::ios::binary);
      file.write(reinterpret_cast<const char*>(data), elements * sizeof(float)); file.close();
      if (!file) throw std::runtime_error("cannot write reference tensor");
    };
    write(".input.f32", backend->input(), kInputElements);
    const auto* raw = backend->infer();
    write(".output.f32", raw, profile.output_elements());
    Config config;
    config.score_threshold = profile.score_threshold; config.iou_threshold = profile.iou_threshold;
    config.pre_nms_limit = profile.pre_nms_limit; config.max_detections = profile.max_detections;
    const auto result = decode(raw, profile.output_elements(), image.cols, image.rows, config, profile);
    cv::FileStorage file(prefix + ".json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!file.isOpened()) throw std::runtime_error("cannot write reference metadata");
    file << "model_id" << profile.model_id << "profile_sha256" << profile.profile_sha256
         << "engine_sha256" << backend->identity().engine_sha256 << "image_sha256" << sha256_file(argv[3])
         << "candidate_count" << int(result.candidate_count) << "nms_candidate_count" << int(result.nms_candidate_count)
         << "nms_count" << int(result.nms_count) << "truncated" << int(result.truncated)
         << "detections" << "[";
    for (const auto& box : result.boxes)
      file << "{" << "class_index" << int(box.class_index) << "label" << profile.classes.at(box.class_index)
           << "score" << box.score << "x1" << box.x1 << "y1" << box.y1 << "x2" << box.x2 << "y2" << box.y2
           << "prediction_index" << int(box.prediction_index) << "}";
    file << "]"; file.release();
    std::cout << result.boxes.size() << " detections, truncated=" << result.truncated << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
