# Repository guidance

This repository owns model contracts, inference, 2D detection and depth-derived
spatial estimates. Camera ownership and acquisition belong to `rgbd_camera`.
Read `README.md` and `docs/interface-v1.md` before changing public behavior.

For shared ownership and system boundaries, read the current
[system architecture](../robot_runtime/docs/research-platform-architecture.md)
in the corresponding runtime checkout. Sections labelled as original proposals
are historical context, not claims about current implementation.

- Keep reusable inference/geometry code behind the thin C++ ROS adapter. Camera
  SDKs do not belong in this repository's implementation dependencies.
- Retain the immutable source RGB-D bundle throughout asynchronous processing.
  Never associate an inference result with the latest unrelated depth.
- Preserve acquisition time, clock, generation and calibration identity. Empty
  detection results, invalid spatial estimates and unavailable processing differ.
- Bound retained work and output size. Surface overload and expiry explicitly.
- Run execution checks only on reserved Jah (`ssh jah`), using isolated files,
  processes, ROS domains and ports as applicable. Local editing, Git operations
  and static inspection are allowed. Preserve installed deployments.
- Record exact software/configuration identities and evidence limits. Synthetic
  or software checks do not establish observed hardware or display performance.
- Follow the user's global workflow preferences; ordinary engineering decisions do
  not require additional approval ceremonies.
- New source is MIT, copyright Paul N. Beuchat. Preserve third-party notices;
  a model descriptor does not license model weights or accelerator artifacts.
- `main` is the verified baseline; `dev` is for development. Work on feature or
  maintenance branches from `dev`. Squash into `dev` after the user's finish
  confirmation. Promote `main` only with separate agreement on a coordinated
  milestone, then synchronize `dev`. Follow the user's global preferences for
  history retention and cleanup. Passing checks alone does not authorize promotion.
- Coordinate public interface/configuration changes with affected consumers.
  Update package/interface versions and `robot_deployment/sources.json` pins
  where required; record compatibility, rebuild requirements and tested revisions.
  Run focused checks on both sides of the changed contract within the authorized scope.

Verification commands and evidence limits are in `docs/verification.md`.
