# Selectable detector profiles

Camera selection and model selection are independent. Both current camera
providers produce the same atomic RGB-D interface; both models run on the same
Jetson TensorRT backend. Select `model_profile` and `engine_path` before starting.
Changing either requires a controlled restart of the detector and its viewer
binding. Startup never exports a model, builds an engine or downloads weights.

| Profile | Classes | Preprocessing | Actual exported output |
| --- | --- | --- | --- |
| `models/cones.json` | yellow_cone, blue_cone | Original bilinear stretch to 640×640 | FP32 `output0 [1,6,8400]` |
| `models/yolov8n-coco80.json` | The 80 COCO labels in checkpoint order | Centred bilinear letterbox to 640×640, padding 114 | FP32 `output0 [1,84,8400]` |

Both inputs are FP32 `images [1,3,640,640]`, RGB NCHW normalized by 255.
Outputs contain centre x/y, width/height in model pixels, then class scores;
there is no objectness channel or exported NMS. The runtime decoder applies
class-aware NMS, source-image clipping and explicit candidate/output limits.
The source coordinate convention and spatial method remain unchanged.

## One shared contract

JSON schema 2 is loaded by the engine builder and runtime through the same C++
validator. It bounds file size and class count, verifies tensor shape/type/names,
requires unique readable class labels and artifact hashes, and rejects unknown
preprocessing/decoder contracts. Only the declared static YOLOv8 tensor family is
supported; an arbitrary ONNX file is not accepted merely because its filename
was selected.

The selected profile supplies preprocessing, decoding, default score/IoU/size
limits and reported model identity. Startup parameters may explicitly override
those detection limits. Every result reports the profile SHA-256 as well as model
and engine SHA-256. Engine manifest schema 2 binds the exact profile bytes.
Changing a profile therefore requires an engine manifest built against that
profile, even if only its provenance text changed.

Legacy schema-1 cone engines remain accepted only for the original cone
model hash, ID, class map, tensor names and stretch contract. They cannot be used
as a general-model engine. The original cone ONNX and inference behavior are
preserved.

For letterboxing, each batch reports the inverse affine transform:
`source_x = model_x * model_to_source_x + model_to_source_offset_x`, likewise for
y. Resized dimensions use nearest-even rounding, padding is centred with the
extra pixel on the right/bottom, and the continuous-coordinate inverse uses the
nominal uniform gain and actual left/top padding. D455 640×480 adds 80 rows above
and below; OAK 640×640 needs no padding. Boxes are clipped before NMS, preserving
the existing detector contract, and padding-only boxes are omitted.

## Pinned general baseline and export

The official checkpoint is
[YOLOv8n from the Ultralytics v8.3.0 asset release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt).
Its SHA-256 is
`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
The inspected ONNX artifact is
`3bd80055fc88933446e2d6d4a7ed346caf65f35bd881fd8b2452924d521bf71d`.
The descriptor contains the complete class catalogue and provenance.

Use an isolated Python 3.12 environment with
[`requirements-export.lock`](../tools/requirements-export.lock). The direct
requirements are also listed in [`requirements-export.txt`](../tools/requirements-export.txt).
The export uses Ultralytics 8.3.0, PyTorch 2.5.1, torchvision 0.20.1, ONNX 1.17.0,
opset 17, batch one, static input, FP32, no graph simplification and no NMS.
Only the wall-clock `date` metadata is removed and metadata keys are ordered to
make the exported artifact reproducible; tensors and graph computation are
unchanged. Export records the full environment and actual I/O tensors.

```bash
python tools/export_yolov8n.py --weights /absolute/yolov8n.pt --output /absolute/new-export
ros2 run rgbd_object_detector build_engine \
  /absolute/new-export/yolov8n.onnx /absolute/yolov8n-coco80.engine \
  /absolute/new-export/yolov8n-coco80.json
ros2 run rgbd_object_detector detector_node --ros-args \
  -p model_profile:=/absolute/new-export/yolov8n-coco80.json \
  -p engine_path:=/absolute/yolov8n-coco80.engine -r frames:=/camera/frames
```

The builder verifies the originating ONNX hash and parsed tensors before building,
then validates the engine through the production loader. The target-specific
manifest records TensorRT/CUDA, architecture, GPU/compute capability, engine and
profile hashes and build settings. Keep engine and same-path `.json` together.
See [NOTICE.md](../NOTICE.md) for upstream licensing; this repository's MIT license
does not relicense model weights or their derived artifacts.

## Saved-image verification

`infer_image` is a non-installed native test executable. It records the production
preprocessed tensor, raw engine output, source-coordinate detections and hashes.
[`verify_reference.py`](../tools/verify_reference.py) compares that output with
ONNX Runtime CPU inference, an independent vectorized decoder, and the pinned
[Ultralytics LetterBox implementation](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/data/augment.py).
The optional checkpoint also checks ONNX export parity against PyTorch.

```bash
python tools/verify_reference.py \
  --native /absolute/workspace/build/rgbd_object_detector/infer_image \
  --engine /absolute/yolov8n-coco80.engine --profile models/yolov8n-coco80.json \
  --onnx /absolute/yolov8n.onnx --checkpoint /absolute/yolov8n.pt \
  --output /absolute/new-reference-results saved-d455.png saved-oak.png blank.png
```

Comparisons report all unmatched detections, source-pixel and confidence errors,
preprocessing differences, raw tensor differences and truncation. Default detected
box/confidence tolerances are 2 pixels and 0.02; they are numerical comparison
criteria, not accuracy bounds. Native synthetic tests separately cover confidence
boundaries, resizing/padding, all-class decoding, empty results, explicit
truncation, invalid depth, source generations and bounded processing.
Neither saved-image parity nor invalid-depth tests establish general-object
spatial quality or live camera/display performance.
