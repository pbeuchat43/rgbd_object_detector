# Selectable-model verification — 27 October 2026

This slice verifies selectable model contracts on the reserved Jetson development
host. It does not establish live viewer performance or general-object spatial
accuracy. The camera source remains the verified `0f4b82958f0d7798f0bf911641d0aa67084cc927`
development baseline. No installed deployment was replaced during these checks.

The native detector contract and composed ROS adapter checks passed with zero
failures/skips. They cover serialized profile rejection, original cone behavior,
80-class decoding, rectangular/square/odd-padding transforms, empty results,
truncation, invalid depth, selected labels/identity, immutable startup selection,
bounded replacement, generation changes and expiry. Three unchanged camera
checks had also passed in the initial isolated workspace build.

Eight saved images were processed by each model: retained D455 and OAK captures,
two ordinary-object reference photos resized to both 640×480 and 640×640, and
black images at both dimensions. Reference photos and black images are fixtures,
not new hardware scene observations. Camera images were retained from the prior
lit stationary-scene campaign. Private imagery and raw tensors stay in the task
archive.

| Comparison | Cone model | YOLOv8n COCO |
| --- | ---: | ---: |
| Images passed | 8/8 | 8/8 |
| Maximum matched source-box coordinate error vs ONNX CPU | 0.0321 px | 0.3309 px |
| Maximum matched confidence error vs ONNX CPU | 0.000892 | 0.002880 |
| Independent decoding of native tensors | Exact within 1e-9 | Exact within 1e-9 |
| Checkpoint-to-ONNX export comparison | Existing ONNX is the reference | Passed all eight images |

The comparison criteria were at most 2 source pixels and 0.02 confidence for
matched TensorRT/ONNX detections, with no unmatched detections. Preprocessing was
checked to 1.3e-7. Full per-image labels, differences, thresholds, source hashes,
artifact identities and outcomes are in [summary.json](summary.json).

The first reference attempt stopped because ONNX Runtime 1.20.1 CPU produced
31 tiny negative cone-class scores on one ordinary-object photo, all exactly
`-5.960464477539063e-08`. The reference-only decoder now records and bounds such
roundoff to 1e-6 before clipping to the mathematical probability range. Larger
violations and nonfinite values still fail; production validation was not relaxed.
The original failed attempt and diagnostic remain in the private archive.

Five negative engine checks passed: wrong selected model, profile hash, model
hash, engine hash, and an attempted general-model legacy manifest. The archived
cone engine remains loadable under its exact legacy contract and produced an
empty result on a black tensor. TensorRT emitted its existing engine portability
warning during that check; this is evidence on this host, not engine portability
acceptance.

The pinned general checkpoint exported to actual FP32 `images [1,3,640,640]` and
`output0 [1,84,8400]`. Two independent executions of the export recipe produced
identical normalized ONNX and descriptor hashes. The new COCO engine was built
on this host, validated through the production loader and exercised with two
independent runtimes plus builder/logger lifetime checks.
[Export provenance](export-provenance.json) and the
[full export dependency lock](../../../tools/requirements-export.lock) record the
build environment. [Source hashes](source-sha256.json) identify the checked
implementation and verification tools.

These checks show numerical agreement for the chosen samples. They do not
measure model precision/recall, surveyed metric depth accuracy, moving scenes,
camera-to-display timing, or sustained rates. Invalid spatial results remain
valid 2D detections with explicit reasons. General-object spatial quality and
live four-combination acceptance remain integration work.
