# Build and verification

Use a prepared Ubuntu 24.04 arm64 / ROS 2 Jazzy host. The verified toolchain is
TensorRT 10.16.2.10, CUDA runtime 13.2 and OpenCV 4.8. This implementation selects
CUDA device 0. Other runtime/hardware combinations need their own evidence.
TensorRT engines are target-specific; startup rejects incompatible manifests.

Execution checks use the designated reserved development host, a separate source,
build and install prefix, a nonzero ROS domain and localhost-only discovery.
Preserve installed services, packages, profiles, power mode and camera ownership.
No command here installs a service or migrates an existing deployment.

## Dependencies

The interface package requires ROS `sensor_msgs` (through the camera interface),
`std_msgs`, `geometry_msgs`, `vision_msgs`, and `rgbd_camera_interfaces` v1.
The implementation additionally requires C++17, CMake >=3.20, `rclcpp`,
`rclcpp_components`, `rcl_interfaces`, OpenSSL, OpenCV core/imgproc, CUDA development
headers/runtime, TensorRT development libraries and the ONNX parser.

The camera baseline used for this milestone is `rgbd_camera` commit
`d956c66d06bec3c30b8b7da6982af60bab0082a8`. Follow that repository's build interface
for OAK acquisition. Source its isolated install before building this overlay.
The detector itself links no DepthAI library. The composed example also requires
the separately installed camera implementation and its SDK library search path.

Where `vision_msgs` is not installed, build it in the isolated workspace from
https://github.com/ros-perception/vision_msgs at Jazzy commit
`adbf56af9f77e3c8ab2552e25dc99e4a8a77fb27`. No system package replacement is required.

```bash
source /opt/ros/jazzy/setup.bash
source /absolute/camera-workspace/install/setup.bash
# Place this repository and the pinned vision_msgs checkout under workspace/src.
cd /absolute/detector-workspace
colcon build --packages-up-to rgbd_object_detector_interfaces --executor sequential
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --packages-select rgbd_object_detector \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

The first command builds interfaces without CUDA, TensorRT or a camera SDK.
`model_profile` selects the validated contract shared by building and runtime.
The default remains `models/cones.json`; [model profiles](model-profiles.md)
documents COCO export, artifact identity and saved-image verification.

## Explicit engine build

Obtain the existing standard one-output cone ONNX separately. Verify SHA-256
`dcf4d911dee975944a3cca895d92803c3181d8dba670759276c4d71901c8040d` and read
`NOTICE.md`. The archived custom three-output ONNX is incompatible.

```bash
ros2 run rgbd_object_detector build_engine \
  /absolute/models/cones.onnx /absolute/models/cones.engine
```

This explicit operation builds an FP16-enabled TensorRT engine with FP32 fallback,
linear FP32 input/output, optimization level 3 and a 1 GiB workspace budget. It
writes a same-path `.json` manifest, then loads the artifact through the production
validator. It refuses to overwrite existing engine/manifest paths. The builder
may take several minutes; startup never performs this build implicitly. Keep
engines and manifests together and rebuild on incompatible runtimes or GPUs.

TensorRT 10.16 marks the FP16 builder flag deprecated. This milestone intentionally
uses its supported TensorRT 10 behavior for the existing FP32 ONNX. Migration to
TensorRT 11/strong typing is not claimed. Warnings remain visible in build logs.

## Run

Standalone, with a separately started camera owner:

```bash
ros2 run rgbd_camera camera_node --ros-args -r __node:=camera \
  -p "device_id:='YOUR_DEVICE_ID'"
ros2 run rgbd_object_detector detector_node --ros-args -r __node:=detector \
  -r frames:=/camera/frames -p engine_path:=/absolute/models/cones.engine
```

Or start the camera and detector together with intra-process communication:

```bash
ros2 launch rgbd_object_detector oak_cones.launch.py \
  device_id:=YOUR_DEVICE_ID engine_path:=/absolute/models/cones.engine
```

The composed topics are `/perception/camera/frames`, `/perception/detector/batches`
and their respective status topics. The camera is the sole hardware owner. Stop
with SIGINT and confirm that the processes exit; a timeout is not permission to
kill unrelated processes or restart the workload automatically.

## Checks

```bash
export ROS_DOMAIN_ID=229 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOG_DIR=/absolute/task/ros-logs
colcon test --packages-select rgbd_object_detector --return-code-on-test-failure \
  --event-handlers console_direct+
colcon test-result --verbose
build/rgbd_object_detector/test_engine /absolute/models/cones.engine models/cones.json
python3 src/rgbd_object_detector/tools/verify_hardware.py \
  --device-id YOUR_DEVICE_ID --engine /absolute/models/cones.engine \
  --test-binaries build/rgbd_object_detector --output /absolute/task/hardware-results
```

The native tests use synthetic output tensors and an injected backend in a real
ROS graph. They check model transforms, NMS, depth math/invalid results, source
validation, exact retention during delayed inference, bounded replacement,
generation changes, expiry, failure observability and cleanup. They establish
software behavior, not hardware accuracy.

The manual `test_engine` checks real inference, binding compatibility and logger
lifetime with two runtimes and a builder. The hardware harness then checks bad
artifact manifests, separate processes, composition and camera restart, and real
inference delayed by a test-only wrapper. That wrapper is neither installed nor
available as a production parameter/component.

The association case joins a bounded sample of actual camera bundles to results
and independently recomputes ROI counts, median depth, MAD and XYZ. This verifies
digital association and arithmetic; it is not a surveyed metric accuracy test.
An empty scene can pass lifecycle/association checks without establishing positive
cone/XYZ observations. The report records positive observations separately.
Add `--require-positive --cases standalone composed association` when visible
cones are prepared; this requires valid XYZ and a positive exact-bundle comparison.

Keep raw reports, scene imagery, device/host identities and preservation audits in
a private development archive. Publish source hashes and a generic verification
summary. Include measurement boundaries: image acquisition age, completed-inference
interval, observer receipt age, process layout, drop counts and product CPU/RSS.
RSS sums across processes include shared pages; they are not PSS. No complete
delivery, hard latency, end-to-end zero-copy, recall or metric accuracy guarantee
follows from these short runs.
