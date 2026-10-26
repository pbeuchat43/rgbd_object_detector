# Third-party references and model provenance

The cone preprocessing, decoding, spatial heuristic and TensorRT binding checks
were developed with reference to `oakd_gpu_cone_detector` at
`5907560cb7a3c5f5df43a9a792243bdee1ce010f`, copyright 2026 The University of
Melbourne, MIT. Its license is retained in `LICENSE.third-party`.

The first model is the existing standard one-output ONNX, SHA-256
`dcf4d911dee975944a3cca895d92803c3181d8dba670759276c4d71901c8040d`.
Its upstream provenance records are under
https://github.com/pbeuchat43/oakd_gpu_cone_detector/tree/5907560cb7a3c5f5df43a9a792243bdee1ce010f/experiments/2026-09-27/cone-model
and `experiments/2026-09-29/provenance` in that repository.

Those records contain conflicting checkpoint AGPL-3.0 metadata and historical
blob MIT attribution, plus a later owner-authorized distribution record. This
repository does not resolve that conflict or assign its MIT license to the
weights. Obtain the existing artifact separately under applicable permissions.
Neither model weights nor TensorRT engines are included in this repository.

The optional general-object baseline is the official Ultralytics YOLOv8n COCO
checkpoint from https://github.com/ultralytics/assets/releases/tag/v8.3.0.
Its checkpoint and normalized ONNX hashes, full 80-class catalogue and export
settings are recorded in `models/yolov8n-coco80.json`. The export/reference tools
use the separately installed Ultralytics 8.3.0 package under its upstream
AGPL-3.0 license: https://github.com/ultralytics/ultralytics/blob/v8.3.0/LICENSE.
No upstream library source or weights are vendored. The MIT license on new
adapter/tool code does not grant rights to redistribute the model, its exported
graph, or its TensorRT engine under MIT; applicable upstream terms remain.

CUDA, TensorRT, OpenCV, ROS 2 and `vision_msgs` retain their own licenses. Building
an engine does not change the model's provenance or distribution conditions.
