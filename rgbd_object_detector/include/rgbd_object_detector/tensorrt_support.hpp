// Copyright (c) 2026 Paul N. Beuchat. MIT License.
#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/utsname.h>

namespace rgbd_object_detector {
class TensorRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
  }
};
// TensorRT shares its registered logger across builders/runtimes. Its lifetime
// must exceed every instance, including multiple components in one process.
TensorRTLogger& shared_tensorrt_logger();
inline void cuda_check(cudaError_t code, const char* operation) {
  if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
struct RuntimeIdentity {
  int tensorrt = 0, tensorrt_build = NV_TENSORRT_BUILD, cuda = 0, major = 0, minor = 0;
  std::string architecture, gpu;
  static RuntimeIdentity inspect() {
    RuntimeIdentity identity;
    identity.tensorrt = getInferLibVersion();
    if (identity.tensorrt != NV_TENSORRT_VERSION)
      throw std::runtime_error("TensorRT headers and loaded library differ; rebuild against the installed runtime");
    cuda_check(cudaSetDevice(0), "select CUDA device 0");
    cuda_check(cudaRuntimeGetVersion(&identity.cuda), "CUDA runtime version");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, 0), "CUDA device properties");
    identity.major = properties.major; identity.minor = properties.minor; identity.gpu = properties.name;
    utsname system{};
    if (uname(&system) != 0) throw std::runtime_error("cannot determine architecture");
    identity.architecture = system.machine;
    return identity;
  }
  void write(cv::FileStorage& out) const {
    out << "tensorrt_runtime" << tensorrt << "tensorrt_build" << tensorrt_build
        << "cuda_runtime" << cuda << "architecture" << architecture << "gpu_name" << gpu
        << "compute_major" << major << "compute_minor" << minor;
  }
  void validate(const cv::FileStorage& in) const {
    if (int(in["tensorrt_runtime"]) != tensorrt || int(in["tensorrt_build"]) != tensorrt_build ||
        int(in["cuda_runtime"]) != cuda || std::string(in["architecture"]) != architecture ||
        std::string(in["gpu_name"]) != gpu || int(in["compute_major"]) != major || int(in["compute_minor"]) != minor)
      throw std::runtime_error("engine runtime/GPU compatibility mismatch; rebuild on this target");
  }
};
}  // namespace rgbd_object_detector
