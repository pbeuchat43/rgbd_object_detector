// Copyright (c) 2026 Paul N. Beuchat. MIT License. See NOTICE.md.
#include "rgbd_object_detector/core.hpp"
#include "rgbd_object_detector/tensorrt_support.hpp"
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace rgbd_object_detector {
TensorRTLogger& shared_tensorrt_logger() {
  static TensorRTLogger logger;
  return logger;
}
namespace {
class TensorRTBackend final : public InferenceBackend {
 public:
  void load(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 16 || file.tellg() > 512 * 1024 * 1024)
      throw std::runtime_error("engine missing or outside 16 bytes..512 MiB");
    std::vector<char> data(size_t(file.tellg()));
    file.seekg(0); file.read(data.data(), data.size());
    if (!file) throw std::runtime_error("cannot read complete engine");
    identity_.engine_sha256 = sha256_bytes(data.data(), data.size());
    identity_.name = "tensorrt_cuda:fp16_with_fp32_fallback";
    cv::FileStorage metadata(path + ".json", cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!metadata.isOpened() || int(metadata["schema_version"]) != 1 ||
        std::string(metadata["model_id"]) != kModelId || std::string(metadata["model_sha256"]) != kModelSha256 ||
        std::string(metadata["engine_sha256"]) != identity_.engine_sha256 ||
        std::string(metadata["precision"]) != "fp16_with_fp32_fallback")
      throw std::runtime_error("engine manifest/model/hash/precision mismatch");
    RuntimeIdentity::inspect().validate(metadata);
    runtime_.reset(nvinfer1::createInferRuntime(shared_tensorrt_logger()));
    if (!runtime_) throw std::runtime_error("TensorRT runtime creation failed");
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) throw std::runtime_error("engine deserialization failed");
    if (engine_->getNbIOTensors() != 2) throw std::runtime_error("expected exactly two engine I/O tensors");
    validate_tensor("images", {1, 3, kModelHeight, kModelWidth}, nvinfer1::TensorIOMode::kINPUT);
    validate_tensor("output0", {1, 6, kPredictions}, nvinfer1::TensorIOMode::kOUTPUT);
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("TensorRT context creation failed");
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create inference stream");
    cuda_check(cudaMallocHost(reinterpret_cast<void**>(&host_input_), kInputElements * sizeof(float)), "allocate pinned input");
    cuda_check(cudaMallocHost(reinterpret_cast<void**>(&host_output_), kOutputElements * sizeof(float)), "allocate pinned output");
    cuda_check(cudaMalloc(&device_input_, kInputElements * sizeof(float)), "allocate device input");
    cuda_check(cudaMalloc(&device_output_, kOutputElements * sizeof(float)), "allocate device output");
    if (!context_->setTensorAddress("images", device_input_) || !context_->setTensorAddress("output0", device_output_))
      throw std::runtime_error("TensorRT tensor binding failed");
  }
  ~TensorRTBackend() override {
    // Also covers partially initialized instances. Cleanup cannot throw from a destructor.
    auto cleanup = [](cudaError_t code, const char* operation) {
      if (code != cudaSuccess) std::cerr << "CUDA cleanup " << operation << ": " << cudaGetErrorString(code) << '\n';
    };
    if (stream_) cleanup(cudaStreamSynchronize(stream_), "synchronize");
    context_.reset();
    if (device_output_) cleanup(cudaFree(device_output_), "free output");
    if (device_input_) cleanup(cudaFree(device_input_), "free input");
    if (host_output_) cleanup(cudaFreeHost(host_output_), "free pinned output");
    if (host_input_) cleanup(cudaFreeHost(host_input_), "free pinned input");
    if (stream_) cleanup(cudaStreamDestroy(stream_), "destroy stream");
    engine_.reset(); runtime_.reset();
  }
  float* input() override { return host_input_; }
  const float* infer() override {
    cuda_check(cudaMemcpyAsync(device_input_, host_input_, kInputElements * sizeof(float), cudaMemcpyHostToDevice, stream_),
               "copy input to device");
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
    cuda_check(cudaMemcpyAsync(host_output_, device_output_, kOutputElements * sizeof(float), cudaMemcpyDeviceToHost, stream_),
               "copy output to host");
    cuda_check(cudaStreamSynchronize(stream_), "complete inference and output transfer");
    return host_output_;
  }
  BackendIdentity identity() const override { return identity_; }
 private:
  void validate_tensor(const char* name, std::initializer_list<int> shape, nvinfer1::TensorIOMode mode) {
    const auto actual = engine_->getTensorShape(name);
    bool matches = actual.nbDims == int(shape.size());
    int index = 0;
    if (matches) for (auto dimension : shape) matches &= actual.d[index++] == dimension;
    if (!matches || engine_->getTensorIOMode(name) != mode ||
        engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
        engine_->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR ||
        engine_->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE)
      throw std::runtime_error(std::string("unsupported tensor shape/type/mode/layout/location: ") + name);
  }
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;
  float* host_input_ = nullptr;
  float* host_output_ = nullptr;
  void* device_input_ = nullptr;
  void* device_output_ = nullptr;
  BackendIdentity identity_;
};
}
std::unique_ptr<InferenceBackend> make_tensorrt_backend(const std::string& engine_path) {
  auto result = std::make_unique<TensorRTBackend>();
  result->load(engine_path);
  return result;
}
}  // namespace rgbd_object_detector
