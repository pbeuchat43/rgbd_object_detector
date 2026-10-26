// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#include "rgbd_object_detector/core.hpp"
#include "rgbd_object_detector/tensorrt_support.hpp"
#include <NvOnnxParser.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <fstream>
#include <memory>

int main(int argc, char** argv) {
  using namespace rgbd_object_detector;
  try {
    if (argc != 3 && argc != 4) throw std::runtime_error("usage: build_engine model.onnx model.engine [model-profile.json]");
    const auto profile = load_model_profile(argc == 4 ? argv[3] :
        ament_index_cpp::get_package_share_directory("rgbd_object_detector") + "/models/cones.json");
    const std::string onnx = argv[1], output = argv[2];
    if (std::filesystem::exists(output) || std::filesystem::exists(output + ".json"))
      throw std::runtime_error("refusing to overwrite an existing engine or manifest");
    if (sha256_file(onnx) != profile.onnx_sha256) throw std::runtime_error("ONNX hash does not match the selected profile");
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
    if (network->getNbInputs() != 1 || network->getNbOutputs() != 1)
      throw std::runtime_error("profile requires exactly one ONNX input and output");
    auto validate = [](const nvinfer1::ITensor& tensor, const std::string& name, std::initializer_list<int> shape) {
      const auto dims = tensor.getDimensions();
      bool valid = name == tensor.getName() && tensor.getType() == nvinfer1::DataType::kFLOAT && dims.nbDims == int(shape.size());
      int index = 0;
      if (valid) for (int dimension : shape) valid &= dims.d[index++] == dimension;
      if (!valid) throw std::runtime_error("ONNX tensor does not match profile: " + name);
    };
    validate(*network->getInput(0), profile.input_name, {1, 3, kModelHeight, kModelWidth});
    validate(*network->getOutput(0), profile.output_name, {1, int(4 + profile.classes.size()), kPredictions});
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
    metadata << "schema_version" << 2 << "model_id" << profile.model_id << "model_sha256" << profile.onnx_sha256
             << "profile_sha256" << profile.profile_sha256
             << "engine_sha256" << sha256_bytes(plan->data(), plan->size())
             << "precision" << "fp16_with_fp32_fallback" << "builder_workspace_mib" << 1024
             << "builder_optimization_level" << 3;
    identity.write(metadata); metadata.release();
    // Validate the artifact through the same path used by the production node.
    const auto backend = make_tensorrt_backend(output, profile);
    std::cout << "Built and validated " << output << " SHA-256 " << backend->identity().engine_sha256 << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
