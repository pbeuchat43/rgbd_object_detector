# OAK to TensorRT to XYZ: observed milestone

The implementation was built and exercised on a reserved NVIDIA Jetson Orin arm64
development host running Ubuntu 24.04 / ROS 2 Jazzy, TensorRT 10.16.2.10, CUDA
runtime 13.2 and OpenCV 4.8. The camera baseline was `rgbd_camera`
`d956c66d06bec3c30b8b7da6982af60bab0082a8`, using its pinned DepthAI 3.9.0 build.
The standard cone ONNX hash is recorded in `models/cones.json`; the target engine
SHA-256 was `35e2b4988f148bb8367c929e16699e0306722290383331baffc239801862ef5e`.
Engines and weights are not distributed by this repository.

`source-sha256.json` identifies the implementation, interfaces, launch file, model
descriptor and verification tools. Raw logs, images, host/device identities,
binary hashes and preservation audits are retained in a private local archive.
No installed deployment was migrated by these checks.

## Software checks

- The interface package built independently of CUDA, TensorRT and the camera SDK.
- Both native tests passed: model/geometry contracts and the ROS adapter with an
  injected backend. These cover bounded replacement, exact retention under delay,
  generation changes, expired results, invalid inputs and terminal backend cleanup.
- The real engine passed tensor/binding checks and inference with two runtimes
  and a builder sharing a process-lifetime logger.
- Deliberately wrong engine hash, model hash and GPU compatibility metadata each
  failed closed. TensorRT's FP16-builder deprecation warning remains documented.
- The shipped `oak_cones.launch.py` entry point produced 10 live batches with
  valid XYZ during a separate startup/shutdown smoke check.

## Live observations

The camera requested 640x640 RGB-D at 10 Hz, IMU disabled; detector defaults were
used. Measurements below include the camera and detector processes but exclude
the Python observer. CPU is a percentage of one core. RSS sums include shared
pages and are not PSS or total GPU memory. Runs are short stationary observations.

| Lit-scene layout | Batches | Output rate | Capture-to-result age p50 / p95 / max | Inference p50 / p95 | Product CPU | Product RSS max |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Separate camera/detector processes | 120 | 9.52 Hz | 48.73 / 50.11 / 51.40 ms | 5.45 / 5.69 ms | 15.0% | 767.4 MiB |
| One container, intra-process input | 120 | 10.00 Hz | 46.28 / 47.89 / 49.24 ms | 5.42 / 5.66 ms | 12.6% | 740.7 MiB |

Both visible yellow and blue cones were detected with valid spatial estimates in
every measured lit-scene batch: 240 valid positions per layout. This is an observed
scene result, not a recall or physical accuracy estimate. A separate association
run matched 10 actual input bundles and independently recomputed all 20 valid
positions, ROI counts, median depths and depth MAD using their original depth and
calibration. Values agreed within 1e-9 metres for the arithmetic comparisons.

An earlier dark-scene run produced explicit empty batches. Composed throughput
was 10.00 Hz, while separate-process throughput was 6.04 Hz. Both runs had 10 Hz
camera publication and no detector admission rejections or pending replacements
in the normal measurement windows. Input sequences were skipped before the
detector callback in separate-process runs; the exact delivery/scheduling loss
point was not isolated. Full-rate external delivery is not established.

Stopping/unloading the camera expired detector readiness without dropping backend
readiness. Reloading the camera produced a new generation and another 40 sustained
batches without old-generation results. A test-only wrapper delayed real inference
by 350 ms: capture continued at 10 Hz, pending replacement increased (79 observed),
five stale items were discarded, and published result age remained below the
configured 500 ms limit. These counters describe the sampled status, not an
end-of-process total. The wrapper is not installed as production functionality.

## Preservation and limits

The installed development release's 1,453 file hashes, configuration hash, selected
deployment, service PID/invocation/start identities and classroom Git state matched
the baseline after verification. Task-owned camera/GPU processes were closed.

No surveyed distance/position ground truth, broad model dataset, physical unplug,
endurance, RealSense, ArUco, browser, runtime deployment or driving verification was
performed. XYZ is a central-ROI median optical-Z surface estimate along the box
centre ray, not a measured physical cone centre. No hard latency, complete delivery,
end-to-end zero-copy or portable TensorRT-engine claim follows from these results.
