# Detector interface v1

`rgbd_object_detector_interfaces` depends only on standard ROS messages and
`rgbd_camera_interfaces`. The C++ processing library is independent of ROS and
camera SDKs. One `rgbd_object_detector::DetectorNode` is independently executable
or composable; camera, inference and spatial logic retain separate ownership.

Optional presentation publishes an exact-source image and atomic `OverlayFrame`;
see [overlays](overlays.md) for association, bounds, status and measurement fields.
The separate `CombinedOverlayNode` joins two original batches with their common
frame and publishes `CombinedOverlayFrame` and `CombinedOverlayStatus` without
changing detector inference or the single-model batch contract.

## Input and association

The `frames` subscription accepts the camera's atomic `Frame` bundle with
best-effort keep-last 1. It requires rectified BGR8 and aligned little-endian
16UC1 optical-Z depth, matching virtual intrinsics and explicit depth units.
Image storage may be padded; dimensions are bounded to 1920x1080. Calibration,
headers, sizes, encodings, timing and pairing metadata are checked before use.

This physical-host implementation requires the same Linux monotonic boot clock.
It rejects remote/unknown clock domains, future/stale captures, duplicate or
regressed sequences, and unsupported geometry. It binds to the first valid
source ID; changing the selected device requires restarting the detector.
Configuration is startup-only. The detector never reconfigures or opens a camera.

One worker owns the inference backend. It retains a const shared pointer to the
complete bundle through preprocessing, inference and spatial estimation. One
pending bundle is replaced by newer input. A generation change clears pending
work and invalidates an in-flight old-generation result before publication.
Monotonic acquisition ordering rejects late old-generation arrivals. Calibration
may change only with a new generation. Output age is checked again after inference.
No queue grows with runtime; middleware/executor queues are additional depth 1.

## Model and spatial semantics

Validated selectable model profiles are defined by `models/cones.json` and
`models/yolov8n-coco80.json`; see [model profiles](model-profiles.md). Engine building
is an explicit offline step, not an automatic node-start operation. Runtime checks
engine SHA-256, originating model SHA-256, TensorRT/CUDA runtime, architecture,
GPU identity/compute capability, and every I/O tensor's shape/type/layout/location.
The backend interface can be replaced without changing acquisition or geometry.

Cone preprocessing retains the original bilinear stretch to 640x640. The COCO
profile uses centred letterboxing with padding 114. Both convert BGR to RGB and
scale to [0,1] NCHW float32. Decoding reverses the declared transform into source
pixel coordinates. Boxes are continuous edge coordinates clipped to [0,width] and
[0,height]; width/height are edge differences. Pixel centres are integer coordinates.
NMS is class-aware, ordered by descending score with deterministic index tie-breaks.
Candidate/output limits and truncation are explicit; no partial batch looks complete.

Spatial estimation uses the central configurable fraction of each clipped box,
rejects the declared invalid code and depths outside configured bounds, and takes
median Z. It projects that Z along the clipped box-centre ray using the matched
rectified K matrix. `central_roi_median_z_box_center` is a region-based surface
estimate, not the physical cone centre. Background, occlusion and box error can
bias it. Minimum sample count and valid fraction are configurable. Depth MAD,
sample count and valid fraction are descriptive quality measures, not uncertainty
or accuracy bounds. Confidence remains the model's class score.

## Output

`~/batches` is an atomic `Batch`: original capture/depth/calibration identity,
model/profile/engine/backend identity, preprocessing scale and offsets, stage timing and a
standard `vision_msgs/Detection2DArray` plus one `Spatial` entry per detection.
Thresholds, region/depth settings and the output age limit travel with each batch.
The spatial entry's `detection_index` is scoped to this batch. Standard detection
IDs remain empty because this component does not track objects. The standard
hypothesis pose/covariance is unavailable (NaN); use the explicit spatial field.

Positions are metres in the unchanged RGB optical frame (x right, y down, z
forward). Acquisition stamps are preserved. Invalid spatial results retain their
2D detection, carry a reason and NaN XYZ, and must not be interpreted as origin.
No orientation, extent, covariance, persistent identity or robot-frame transform
is inferred. A successfully processed empty batch replaces prior detections.

`~/status` emits every 200 ms with a 1 s expiry in the declared local clock. It
separates backend readiness, input progress and successful processing, and reports
received/rejected/replaced/stale/superseded/processed counts and queue occupancy.
When no recent successful acquisition remains, readiness becomes unavailable.
Backend errors are terminal for that node instance and release its GPU resources;
the node remains observable in error until explicitly restarted/unloaded.

Latency stages are host monotonic intervals: preprocessing, GPU submission through
completed D2H copy, and decoding/spatial estimation. Total processing excludes
queue wait; source age at completion includes it. These are observed intervals,
not hard deadlines. CUDA synchronization cannot be forcibly preempted by this node.

## Parameters

All are read-only after construction. `model_profile` defaults to the installed
`models/cones.json`. `engine_path` is required; its manifest is the same path plus
`.json`. Detection defaults come from the selected validated profile; both current
profiles declare `score_threshold=0.5`, `iou_threshold=0.5`,
`pre_nms_limit=1000` (1..8400), `max_detections=20` (1..100),
`roi_fraction=0.5`, `min_depth_m=0.1`, `max_depth_m=5.0`,
`min_valid_pixels=5`, `min_valid_fraction=0.05`, `max_frame_age_ms=500`
(10..2000). `use_sim_time=true` is unsupported for this physical-host adapter.
No test-delay or synthetic-detection parameter exists in the production node.

The initial spatial contract accepts a finite declared depth scale from
1e-9 through 1 metre per raw unit. Additional formats, distortion models,
unaligned depth and cross-host timing need explicit extensions and evidence.
