// Synthetic math/model contract checks. No model engine, GPU or camera is opened.
#include "rgbd_object_detector/core.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace rgbd_object_detector;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F action, const char* message) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error(message);
}
void prediction(std::vector<float>& out, int index, float x, float y, float w, float h,
                float yellow, float blue) {
  const float values[]{x,y,w,h,yellow,blue};
  for (int channel = 0; channel < 6; ++channel) out[channel * kPredictions + index] = values[channel];
}
int main() {
  try {
    Config config; config.validate();
    auto invalid = config; invalid.roi_fraction = NAN;
    rejects([&] { invalid.validate(); }, "nonfinite setting accepted");
    invalid = config; invalid.min_depth_m = invalid.max_depth_m;
    rejects([&] { invalid.validate(); }, "unordered depth bounds accepted");
    invalid = config; invalid.max_frame_age_ms = 2001;
    rejects([&] { invalid.validate(); }, "unbounded age accepted");

    // Padded, nonsquare input exercises BGR order, scaling, stride and resize.
    std::vector<uint8_t> bgr(16 * 2, 199);
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 4; ++x) {
      bgr[y * 16 + x * 3] = 10; bgr[y * 16 + x * 3 + 1] = 20; bgr[y * 16 + x * 3 + 2] = 30;
    }
    const auto original = bgr;
    std::vector<float> input(kInputElements);
    preprocess({bgr.data(), bgr.size(), 4, 2, 16}, input.data(), input.size());
    check(bgr == original, "preprocess mutated shared source");
    check(std::abs(input[0] - 30.0/255) < 1e-6 && std::abs(input[640*640] - 20.0/255) < 1e-6 &&
          std::abs(input.back() - 10.0/255) < 1e-6, "BGR/RGB scaling/stride/layout wrong");
    rejects([&] { preprocess({bgr.data(), bgr.size()-1, 4, 2, 16}, input.data(), input.size()); }, "short image accepted");

    std::vector<float> output(kOutputElements, 0);
    prediction(output, 0, 320, 320, 160, 160, .9F, .1F);
    prediction(output, 1, 320, 320, 160, 160, .8F, .1F); // suppressed, same class
    prediction(output, 2, 320, 320, 160, 160, .1F, .85F); // retained, different class
    prediction(output, 3, 100, 100, 40, 40, .5F, .1F); // inclusive threshold
    auto decoded = decode(output.data(), output.size(), 1280, 480, config);
    check(decoded.candidate_count == 4 && decoded.nms_count == 3 && decoded.boxes.size() == 3 &&
          !decoded.truncated, "class-aware NMS or inclusive threshold wrong");
    const auto box = decoded.boxes[0];
    check(box.x1 == 480 && box.x2 == 800 && box.y1 == 180 && box.y2 == 300,
          "model-to-source anisotropic scale wrong");
    config.max_detections = 1;
    decoded = decode(output.data(), output.size(), 640, 640, config);
    check(decoded.truncated && decoded.boxes.size() == 1 && decoded.nms_count == 3, "output truncation hidden");
    config.pre_nms_limit = 2;
    decoded = decode(output.data(), output.size(), 640, 640, config);
    check(decoded.truncated && decoded.candidate_count == 4 && decoded.nms_candidate_count == 2,
          "candidate truncation hidden");
    config = Config{};
    std::fill(output.begin(), output.end(), 0);
    prediction(output, 0, 0, 0, 40, 40, .9F, .1F);
    decoded = decode(output.data(), output.size(), 640, 640, config);
    check(decoded.boxes[0].x1 == 0 && decoded.boxes[0].x2 == 20, "box clipping wrong");
    output[4*kPredictions] = 1.1F;
    rejects([&] { decode(output.data(), output.size(), 640, 640, config); }, "invalid score accepted");
    output[4*kPredictions] = NAN;
    rejects([&] { decode(output.data(), output.size(), 640, 640, config); }, "nonfinite output accepted");
    std::fill(output.begin(), output.end(), 0);
    check(decode(output.data(), output.size(), 640, 640, config).boxes.empty(), "empty batch fabricated detections");

    // A constant plane with invalid pixels and an outlier checks units/median/projection.
    std::vector<uint8_t> storage(20 * 8, 0);
    auto set_depth = [&](int x, int y, uint16_t raw) { storage[y*20+x*2] = raw & 255; storage[y*20+x*2+1] = raw >> 8; };
    for (int y = 0; y < 8; ++y) for (int x = 0; x < 8; ++x) set_depth(x,y,2000);
    DepthView depth{{storage.data(), storage.size(), 8, 8, 20}, {100,0,4,0,100,4,0,0,1}, .001, 0};
    Box region{2,2,8,8,.9,0,0};
    set_depth(4,4,0); set_depth(5,5,5000);
    auto spatial = estimate(region, depth, config);
    check(spatial.valid && spatial.roi_pixels == 9 && spatial.valid_pixels == 8 && spatial.xyz[2] == 2 &&
          std::abs(spatial.xyz[0] - .02) < 1e-12 && std::abs(spatial.xyz[1] - .02) < 1e-12 &&
          spatial.depth_mad_m == 0, "depth units/region/median/projection wrong");
    config.min_valid_fraction = 1;
    spatial = estimate(region, depth, config);
    check(!spatial.valid && spatial.reason == "low_valid_fraction" && std::isnan(spatial.xyz[0]), "low quality invented XYZ");
    std::fill(storage.begin(), storage.end(), 0);
    spatial = estimate(region, depth, config);
    check(!spatial.valid && spatial.reason == "insufficient_depth" && std::isnan(spatial.xyz[2]), "invalid depth became origin");
    depth.unit_m = NAN;
    rejects([&] { estimate(region, depth, config); }, "nonfinite depth unit accepted");
    check(sha256_bytes("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 wrong");
    std::cout << "detector core checks passed (synthetic model outputs and geometry)\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
