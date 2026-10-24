// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#include "rgbd_object_detector/core.hpp"
#include "rgbd_object_detector/tensorrt_support.hpp"
#include <NvOnnxParser.h>
#include <filesystem>
#include <fstream>
#include <memory>

int main(int argc, char** argv) {
  using namespace rgbd_object_detector;
  try {
    if (argc != 3) throw std::runtime_error("usage: build_engine /absolute/cones.onnx /absolute/cones.engine");
    const std::string onnx = argv[1], output = argv[2];
    if (std::filesystem::exists(output) || std::filesystem::exists(output + ".json"))
      throw std::runtime_error("refusing to overwrite an existing engine or manifest");
    if (sha256_file(onnx) != kModelSha256) throw std::runtime_error("ONNX hash does not match the cone profile");
    const auto identity = RuntimeIdentity::inspect();
    auto& logger = shared_tensorrt_logger();
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("TensorRT builder creation failed");
    builder->setMaxThreads(2);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) throw std::runtime_error("TensorRT network creation failed");
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser || !parser->parseFromFile(onnx.c_str(), int(nvinfer1::ILogger::Severity::kWARNING)))
      throw std::runtime_error("ONNX parsing failed");
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) throw std::runtime_error("TensorRT builder configuration failed");
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1024ULL * 1024 * 1024);
    config->setBuilderOptimizationLevel(3);
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) throw std::runtime_error("TensorRT engine build failed");
    {
      std::ofstream file(output, std::ios::binary);
      file.write(static_cast<const char*>(plan->data()), plan->size());
      file.close();
      if (!file) throw std::runtime_error("engine write failed");
    }
    cv::FileStorage metadata(output + ".json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!metadata.isOpened()) throw std::runtime_error("engine manifest write failed");
    metadata << "schema_version" << 1 << "model_id" << kModelId << "model_sha256" << kModelSha256
             << "engine_sha256" << sha256_bytes(plan->data(), plan->size())
             << "precision" << "fp16_with_fp32_fallback" << "builder_workspace_mib" << 1024
             << "builder_optimization_level" << 3;
    identity.write(metadata); metadata.release();
    // Validate the artifact through the same path used by the production node.
    const auto backend = make_tensorrt_backend(output);
    std::cout << "Built and validated " << output << " SHA-256 " << backend->identity().engine_sha256 << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
