# CaptureDetections migration (0.2.0)

Copyright (c) 2026 Paul N. Beuchat. MIT License.

This coordinated naming change replaces `Batch` with `CaptureDetections`: one
model's results for one exact capture. Detector and interface packages are 0.2.0.
The payload fields, ordering and semantic `schema_version = 1` are unchanged.

| Before | After |
| --- | --- |
| `rgbd_object_detector_interfaces/msg/Batch` | `rgbd_object_detector_interfaces/msg/CaptureDetections` |
| C++ `msg/batch.hpp`, `msg::Batch`; Python `msg.Batch` | `msg/capture_detections.hpp`, `msg::CaptureDetections`; `msg.CaptureDetections` |
| Detector `~/batches` | `~/capture_detections` |
| Combined input `first_batches` / `second_batches` | `first_capture_detections` / `second_capture_detections` |
| `OverlayFrame.batch` (`Batch`) | `OverlayFrame.capture_detections` (`CaptureDetections`) |
| `CombinedOverlayFrame.batches` (`Batch[2]`) | `CombinedOverlayFrame.capture_detections` (`CaptureDetections[2]`) |
| `make_batch`, `draw_batch`, `receive_batch` | `make_capture_detections`, `draw_capture_detections`, `receive_capture_detections` |
| Message-related `batch` / `batches` variables, queues, publishers and subscriptions | `capture_detections` names; `model_results` for the renderer's per-model array |
| Probe `batches.csv`, observer CSV `batches` count column | `capture_detections.csv`, `capture_detections` count column |
| Hardware report `batch_count`, `empty_batches`, `batch_rate_hz`, `batches` | `capture_detections_count`, `empty_capture_detections`, `capture_detections_rate_hz`, `capture_detections` |

Under the deployment namespace, model outputs are
`detector_cones/capture_detections` and `detector_coco/capture_detections`.
The combined node remaps its first input to cones and its second input to COCO.
The standalone example retains the node name `detector`. Status and image topic
names, QoS and parameters are unchanged.

Capture, model, calibration and depth-pair identities remain attached to every
result. Acquisition stamps, spatial indices/validity, explicit fresh empty
results, bounded work and exact overlay association retain their existing
semantics. This adds no tracking, cross-model merged result or observation layer.
Model tensor batch-one settings, exporter arguments, weights and engines are
unchanged; the rename alone does not require regenerating an engine.

## Coordinated rebuild

This is a ROS source, generated-binding and wire compatibility break. The old
type is removed, without a compatibility publisher or alias. Remapping only a
topic cannot convert the old type. `OverlayFrame` and `CombinedOverlayFrame`
retain their type names but change their embedded types/fields and type hashes.

Build into a clean isolated prefix: generate `rgbd_object_detector_interfaces`,
then rebuild `rgbd_object_detector`, including both presentation components,
all C++ clients and any consumers of either atomic overlay. Regenerate Python
bindings and update imports, fields and subscriptions. Do not layer a partially
rebuilt consumer over stale detector libraries or generated message files.

Workspace consumers are `robot_deployment`'s launcher and individual/combined
integration probes, and `rgbd_camera/tools/rate_probe`. Rebuild both probe
projects against 0.2.0 and update CSV/report readers. Deployment pins select the
coordinated revisions. The camera provider/interface and `robot_runtime` require
no implementation change: runtime consumes standard images and unchanged status
messages. Student/external subscribers, visualizers, launch remaps, recording
tools and bag readers using the old type or either atomic overlay must migrate
before use with the new graph; no external consumer is claimed tested here.

Old bag data and evidence retain their original message bindings and software
identities. Use matching historical software to read them; do not relabel their
type, topic, hashes or measured results. `docs/evidence`, the first milestone
record, archived task scripts and source manifests remain historical. They do
not establish hardware acceptance of 0.2.0. Deployment and new hardware tests
require separate authorization.
