#!/usr/bin/env python3
"""Compare native preprocessing/decoding and TensorRT with saved-image ONNX reference inference.

Requires the isolated requirements-export.txt environment and the native
infer_image test executable. Inputs and raw tensors stay in the private output
directory. Synthetic depth/empty/truncation behavior is tested by native tests;
this tool establishes saved-image numerical parity, not detector accuracy.
"""
# Copyright (c) 2026 Paul N. Beuchat. MIT License.
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(raw, profile, width, height, *, score_roundoff=0.0):
    """Independent vectorized reference for the declared clipped-box NMS contract."""
    import numpy as np
    values = np.asarray(raw[0], dtype=np.float64).T
    if not np.isfinite(values).all() or not ((values[:, 4:] >= -score_roundoff) & (values[:, 4:] <= 1 + score_roundoff)).all():
        raise ValueError("invalid reference tensor")
    # ORT 1.20.1's CPU sigmoid can return -5.96e-8. Admit only explicit,
    # measured floating-point roundoff in this reference, never in production.
    values[:, 4:] = np.clip(values[:, 4:], 0, 1)
    labels = values[:, 4:].argmax(axis=1)
    scores = values[np.arange(len(values)), labels + 4]
    xyxy = np.column_stack((values[:, 0] - values[:, 2] / 2, values[:, 1] - values[:, 3] / 2,
                           values[:, 0] + values[:, 2] / 2, values[:, 1] + values[:, 3] / 2))
    if profile["resize"]["mode"] == "letterbox":
        gain = min(640 / width, 640 / height)
        left, top = (640 - round(width * gain)) // 2, (640 - round(height * gain)) // 2
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / gain
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / gain
    else:
        xyxy[:, [0, 2]] *= width / 640
        xyxy[:, [1, 3]] *= height / 640
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
    settings = profile["defaults"]
    admitted = np.flatnonzero((scores >= settings["score_threshold"]) & (values[:, 2] > 0) &
        (values[:, 3] > 0) & (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1]))
    ordered = admitted[np.lexsort((admitted, -scores[admitted]))]
    pending = ordered[:settings["pre_nms_limit"]]
    retained = []
    while len(pending):
        chosen, rest = int(pending[0]), pending[1:]
        retained.append(chosen)
        if not len(rest):
            break
        a, b = xyxy[chosen], xyxy[rest]
        intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2])), axis=1)
        area = np.prod(a[2:] - a[:2]) + np.prod(b[:, 2:] - b[:, :2], axis=1) - intersection
        suppress = (labels[rest] == labels[chosen]) & (intersection / area > settings["iou_threshold"])
        pending = rest[~suppress]
    results = []
    for index in retained[:settings["max_detections"]]:
        results.append(dict(class_index=int(labels[index]), label=profile["classes"][labels[index]],
            score=float(scores[index]), prediction_index=index,
            **dict(zip(("x1", "y1", "x2", "y2"), xyxy[index].tolist()))))
    return dict(detections=results, candidate_count=len(admitted), nms_candidate_count=min(len(ordered), settings["pre_nms_limit"]),
        nms_count=len(retained), truncated=int(len(ordered) > settings["pre_nms_limit"] or len(retained) > settings["max_detections"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional pinned PyTorch checkpoint for additional ONNX export parity")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-tolerance", type=float, default=0.02)
    parser.add_argument("--box-tolerance-px", type=float, default=2.0)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(args.output / "tool-state")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics.data.augment import LetterBox
    torch.set_num_threads(2)
    profile = json.loads(args.profile.read_text())
    if digest(args.onnx) != profile["onnx_sha256"]:
        raise ValueError("reference ONNX hash does not match profile")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=session_options, providers=["CPUExecutionProvider"])
    checkpoint_model = None
    if args.checkpoint:
        if digest(args.checkpoint) != profile["provenance"]["checkpoint_sha256"]:
            raise ValueError("reference checkpoint hash does not match profile")
        from ultralytics import YOLO
        checkpoint_model = YOLO(str(args.checkpoint)).model.float().eval()
        checkpoint_model.fuse(verbose=False)
    results = []
    for index, path in enumerate(args.images):
        prefix = args.output / f"image-{index:03d}"
        subprocess.run([str(args.native), str(args.engine), str(args.profile), str(path), str(prefix)], check=True)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        if profile["resize"]["mode"] == "letterbox":
            resized = LetterBox(new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True)(image=image)
        else:
            resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        prepared = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
        actual_input = np.fromfile(str(prefix) + ".input.f32", np.float32).reshape(1, 3, 640, 640)
        input_error = float(np.max(np.abs(prepared - actual_input)))
        reference_raw = session.run([profile["output"]["name"]], {profile["input"]["name"]: prepared})[0]
        expected_shape = (1, len(profile["classes"]) + 4, 8400)
        if reference_raw.shape != expected_shape:
            raise ValueError(f"unexpected actual reference output {reference_raw.shape}")
        actual_raw = np.fromfile(str(prefix) + ".output.f32", np.float32).reshape(expected_shape)
        actual = json.loads(Path(str(prefix) + ".json").read_text())
        decoded_actual = decode(actual_raw, profile, width, height)
        decoder_equal = all(actual[key] == decoded_actual[key] for key in ("candidate_count", "nms_candidate_count", "nms_count", "truncated"))
        decoder_equal &= len(actual["detections"]) == len(decoded_actual["detections"])
        for a, b in zip(actual["detections"], decoded_actual["detections"]):
            decoder_equal &= all(a[key] == b[key] for key in ("class_index", "label", "prediction_index"))
            decoder_equal &= all(abs(a[key] - b[key]) < 1e-9 for key in ("score", "x1", "y1", "x2", "y2"))
        reference = decode(reference_raw, profile, width, height, score_roundoff=1e-6)
        unmatched = list(actual["detections"])
        comparisons = []
        for expected in reference["detections"]:
            candidates = [d for d in unmatched if d["class_index"] == expected["class_index"]]
            if not candidates:
                comparisons.append(dict(label=expected["label"], matched=False))
                continue
            closest = min(candidates, key=lambda d: max(abs(d[key] - expected[key]) for key in ("x1", "y1", "x2", "y2")))
            unmatched.remove(closest)
            error = max(abs(closest[key] - expected[key]) for key in ("x1", "y1", "x2", "y2"))
            score_error = abs(closest["score"] - expected["score"])
            comparisons.append(dict(label=expected["label"], matched=error <= args.box_tolerance_px and score_error <= args.score_tolerance,
                                    box_max_error_px=error, score_error=score_error))
        export_parity = None
        if checkpoint_model is not None:
            with torch.inference_mode():
                torch_raw = checkpoint_model(torch.from_numpy(prepared))[0].numpy()
            export_parity = dict(box_max_error_px=float(np.max(np.abs(torch_raw[:, :4] - reference_raw[:, :4]))),
                                 score_max_error=float(np.max(np.abs(torch_raw[:, 4:] - reference_raw[:, 4:]))))
            export_parity["passed"] = export_parity["box_max_error_px"] <= 0.02 and export_parity["score_max_error"] <= 0.0001
        passed = input_error <= 1.3e-7 and decoder_equal and not unmatched and all(row["matched"] for row in comparisons)
        passed &= export_parity is None or export_parity["passed"]
        record = dict(image=str(path), image_sha256=digest(path), width=width, height=height,
            preprocessing_max_error=input_error, native_decoder_equal=bool(decoder_equal),
            raw_box_max_error_px=float(np.max(np.abs(actual_raw[:, :4] - reference_raw[:, :4]))),
            raw_score_max_error=float(np.max(np.abs(actual_raw[:, 4:] - reference_raw[:, 4:]))),
            reference_score_range=[float(reference_raw[:, 4:].min()), float(reference_raw[:, 4:].max())],
            reference_score_roundoff_count=int(((reference_raw[:, 4:] < 0) | (reference_raw[:, 4:] > 1)).sum()),
            reference=reference, native=actual, comparisons=comparisons, unmatched_native=unmatched,
            pytorch_onnx_parity=export_parity, passed=bool(passed))
        results.append(record)
        Path(str(prefix) + ".comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    report = dict(model_id=profile["model_id"], profile_sha256=digest(args.profile), engine_sha256=digest(args.engine),
        onnx_sha256=digest(args.onnx), native_sha256=digest(args.native), script_sha256=digest(__file__),
        score_tolerance=args.score_tolerance, box_tolerance_px=args.box_tolerance_px,
        reference_score_roundoff_tolerance=1e-6,
        cases=results, passed=all(row["passed"] for row in results))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(model_id=report["model_id"], cases=len(results), passed=report["passed"])))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
