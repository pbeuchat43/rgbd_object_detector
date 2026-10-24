# Repository guidance

This repository owns model contracts, inference, 2D detection and depth-derived
spatial estimates. Camera ownership and acquisition belong to `rgbd_camera`.
Read `README.md` and `docs/interface-v1.md` before changing public behavior.

- Keep reusable inference/geometry code behind the thin C++ ROS adapter. Camera
  SDKs do not belong in this repository's implementation dependencies.
- Retain the immutable source RGB-D bundle throughout asynchronous processing.
  Never associate an inference result with the latest unrelated depth.
- Preserve acquisition time, clock, generation and calibration identity. Empty
  detection batches, invalid spatial estimates and unavailable processing differ.
- Bound retained work and output size. Surface overload and expiry explicitly.
- Run execution checks on the designated reserved development host using isolated resources. Preserve
  installed deployments and record actual software identity and evidence limits.
- Follow the user's Solo/workflow preferences; ordinary engineering decisions do
  not require additional approval ceremonies.
- New source is MIT, copyright Paul N. Beuchat. Preserve third-party notices;
  a model descriptor does not license model weights or accelerator artifacts.
- `main` is the verified baseline and `dev` development. Preserve development
  history locally before consolidating and publishing a verified milestone.
