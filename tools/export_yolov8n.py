#!/usr/bin/env python3
"""Export the pinned official YOLOv8n COCO checkpoint, inspect it, and record provenance.

Run only in an isolated environment from requirements-export.txt. Does not fetch
weights, install packages, build an engine, or start camera/runtime workloads.
"""
# Copyright (c) 2026 Paul N. Beuchat. MIT License.
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
WEIGHTS_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
VERSIONS = {"ultralytics": "8.3.0", "torch": "2.5.1", "torchvision": "0.20.1",
            "onnx": "1.17.0", "onnxruntime": "1.20.1", "numpy": "1.26.4", "opencv-python": "4.10.0.84"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    args = parser.parse_args()
    if sha256(args.weights) != WEIGHTS_SHA256:
        raise ValueError("checkpoint does not match the pinned official YOLOv8n artifact")
    actual_versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if actual_versions != VERSIONS:
        raise ValueError(f"export dependency mismatch: {actual_versions}")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(args.output / "tool-state")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    import onnx
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    checkpoint = args.output / "yolov8n.pt"
    shutil.copyfile(args.weights, checkpoint)
    model = YOLO(str(checkpoint), task="detect")
    if set(model.names) != set(range(80)) or model.task != "detect":
        raise ValueError("checkpoint is not the expected 80-class detection model")
    export_args = dict(format="onnx", imgsz=640, batch=1, dynamic=False, half=False,
                       simplify=False, opset=17, nms=False, device="cpu")
    exported = Path(model.export(**export_args))
    graph = onnx.load(exported)
    # Remove only wall-clock export metadata so identical exports can have the
    # same artifact hash. The execution/provenance record remains separate.
    metadata = {entry.key: entry.value for entry in graph.metadata_props if entry.key != "date"}
    del graph.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(graph)

    def tensor(value):
        shape = [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("export must use FP32 input and output")
        return dict(name=value.name, dtype="float32", shape=shape)

    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError("expected exactly one input and one output")
    input_tensor, output_tensor = tensor(graph.graph.input[0]), tensor(graph.graph.output[0])
    if input_tensor != dict(name="images", dtype="float32", shape=[1, 3, 640, 640]):
        raise ValueError(f"unexpected actual input: {input_tensor}")
    if output_tensor != dict(name="output0", dtype="float32", shape=[1, 84, 8400]):
        raise ValueError(f"unexpected actual output: {output_tensor}")
    onnx.save(graph, exported)
    profile = dict(schema_version=2, model_id="yolov8n-coco80-ultralytics-8.3.0-opset17",
        onnx_sha256=sha256(exported), input=input_tensor, output=output_tensor,
        classes=[model.names[index] for index in range(80)],
        preprocessing="bgr8_to_rgb_nchw_div255_bilinear",
        resize=dict(mode="letterbox", padding_value=114), decoder="yolov8_xywh_class_scores",
        defaults=dict(score_threshold=0.5, iou_threshold=0.5, pre_nms_limit=1000, max_detections=20),
        provenance=dict(checkpoint_url=WEIGHTS_URL, checkpoint_sha256=WEIGHTS_SHA256,
            exporter="ultralytics==8.3.0", opset=17, dynamic=False, simplify=False,
            half=False, nms=False, batch=1, export_date_metadata="removed for reproducibility"),
        licensing="Ultralytics AGPL-3.0 / applicable upstream terms; see NOTICE.md. No weights distributed here.")
    profile_path = args.output / "yolov8n-coco80.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    provenance = dict(schema_version=1, checkpoint_sha256=WEIGHTS_SHA256,
        onnx_sha256=sha256(exported), profile_sha256=sha256(profile_path),
        export_script_sha256=sha256(__file__), versions=actual_versions,
        python=sys.version, platform=platform.platform(), export_args=export_args,
        input=input_tensor, output=output_tensor, classes=profile["classes"],
        opsets=[dict(domain=entry.domain, version=entry.version) for entry in graph.opset_import])
    (args.output / "export-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output / "requirements.lock").write_text(frozen)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
