# First detector milestone

Implement OAK acquisition through the published camera v1 contract, host TensorRT
cone inference, and exact-bundle depth-derived XYZ. Keep the current camera and
runtime baselines intact unless a verified interface defect requires a scoped fix.

The first detector is one composable/standalone C++ node. A replaceable inference
adapter, model preprocessing/decoder, and reusable spatial estimator are separate
code boundaries. A second ROS spatial node would add image retention and joining
without a current deployment need. The public output leaves those boundaries
available for later implementations, including on-device inference.

Acceptance:

- The interface package builds independently of TensorRT and camera SDKs.
- Native tests cover source-coordinate mapping, class-aware NMS and truncation,
  valid/invalid depth, malformed input, generation changes during inference,
  stale results, bounded replacement, backend failure and cleanup.
- A target-built engine validates the pinned ONNX hash, runtime compatibility,
  tensor names/shapes/types/layouts and engine hash. A bad artifact fails closed.
- Isolated hardware runs produce successful batches from real OAK frames in
  standalone and composed layouts. Empty batches remain explicit. If cones are
  visible, record observed class/score/XYZ without asserting surveyed accuracy.
- Deliberately slow processing keeps one active and one pending bundle, discards
  obsolete results, and permits capture to continue. Restart does not associate
  frames across source generations. Input loss expires detector readiness.
- Record source/artifact identities, latency/rate distributions, drops, CPU/RSS,
  process layout and limits. No numerical throughput/accuracy guarantee is assumed.
- Preserve installed dev17 files/configuration, service identities and classroom
  Git state; close only this task's processes and release the reservation.

RealSense, ArUco, browser video, runtime deployment, tracking, frame transforms and
vehicle control are outside this milestone. Physical depth accuracy, model recall
and endurance need separate datasets/setups and are not established by fixtures.
