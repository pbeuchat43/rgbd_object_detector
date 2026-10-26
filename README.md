# rgbd_object_detector

Composable C++ RGB-D object detection for the research robot platform. Select a
validated model profile independently of the camera, then derive spatial
estimates from the exact RGB-D bundle used for inference. Acquisition remains in
the separately installed `rgbd_camera` provider.

The detector supports standalone execution and ROS 2 composition. One worker owns
inference and retains its input bundle; one replaceable pending bundle bounds
backlog. Inference and spatial estimation are separate reusable library functions
behind one ROS component. They do not require separate ROS nodes or image copies.

See [the interface](docs/interface-v1.md), [milestone acceptance](docs/milestone.md),
and [build and verification](docs/verification.md). The
[observed milestone results](docs/evidence/2026-10-24-oak-tensorrt/README.md) separate
software checks from live cone observations and record their limits. Version 0.1.0
is a development milestone, not a stable platform or accuracy claim.

The [model profile guide](docs/model-profiles.md) covers the existing
[cone model](models/cones.json) and pinned [YOLOv8n COCO model](models/yolov8n-coco80.json).
Profiles pin preprocessing, tensor contracts, classes and artifact identity.
Model weights and target-specific engines are not distributed here.
See [NOTICE.md](NOTICE.md) for provenance and licensing qualifications.

Canonical repository: https://github.com/pbeuchat43/rgbd_object_detector.
New code is MIT, copyright 2026 Paul N. Beuchat.
