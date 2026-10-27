# Exact-capture overlays

The detector's optional presentation worker draws on an owned copy of the RGB
image retained for that inference. It never looks up a newer image by arrival
time. `publish_overlay` defaults to false; `overlay_spatial` adds optical Z for
valid spatial estimates and `2D only` otherwise. Both are startup-only parameters.

The presentation library is separate from the reusable inference/geometry core.
The worker retains at most one active and one pending frame/result pair; a new
pending pair replaces the previous one. Rendering and ROS publication run outside
the inference thread and its queue lock. Outputs use volatile, best-effort,
depth-one QoS. Work is skipped without subscribers. Source generation changes
invalidate pending work, and active work is checked for supersession and expiry
before publication. An overlay failure is reported independently and does not
terminate inference.

- `~/overlay/image` (`sensor_msgs/Image`) contains BGR8 pixels with the original
  RGB header and acquisition stamp. It is suitable for a ROS-image video encoder.
- `~/overlay/frame` (`rgbd_object_detector_interfaces/OverlayFrame`) atomically
  includes those pixels, the complete detection batch, and monotonic rendering
  start/completion times. Its batch preserves source/generation/sequence,
  calibration, model/profile/engine identity, and acquisition clock metadata.

Boxes and labels use source-image coordinates and include class names and
confidence. Valid spatial estimates are green; invalid estimates retain an amber
2D box. A model/capture banner reports detection count and explicit `TRUNCATED`
state. A fresh empty batch produces a fresh image without previous boxes.

Status exposes enabled/state/detail, active/pending work, rendered/replaced/stale/
superseded/no-consumer/failure counts and the last acquisition/render duration.
The image header alone does not contain the full capture identity. For consumers
that need machine-readable identity, use the atomic frame; do not join a separate
detection stream to whichever decoded browser frame happens to be current.

## Combined presentation node

`rgbd_object_detector::CombinedOverlayNode` is a separate composable ROS node,
also available as `combined_overlay_node`. It subscribes to `frames`,
`first_batches` and `second_batches`. It owns no inference backend or camera.
Load it with the camera and both detectors using intra-process communication to
share immutable input bundles. Rendering copies the source RGB pixels once.

The node joins the original frame and both complete batches by capture sequence
within the current source/generation. Before rendering it requires full capture,
calibration, RGB header/geometry, depth pairing and camera-backend equality.
Each input keeps its own model/profile/engine identity and spatial indices. A
valid empty batch completes a pair; a missing batch does not. Model identities
are fixed after their first accepted batch and require restarting the node to
change. Only camera frames can advance the generation; late older generations
cannot roll it back. Sequences must advance within each input stream, while
arrival order between the three streams is unrestricted after source binding.

`join_capacity` defaults to 8 (2..32). It bounds the total unmatched capture
entries, including entries containing only a batch. The oldest sequence is
evicted when full. There is additionally one active and one replaceable pending
render. `max_frame_age_ms` defaults to 500 (10..2000); the stricter limit from
either batch also applies. Expiry is checked before and after rendering, and
unmatched entries expire on the 200 ms status timer. Generation changes discard
old work. Rendering and publication run on a dedicated thread outside the join
mutex. No subscribers means no retained render/join work. These are freshness
checks, not a hard real-time publication deadline.

- `~/overlay/image` is the combined BGR8 image with its original RGB header.
- `~/overlay/frame` is `CombinedOverlayFrame`: two unchanged batches, the image,
  and monotonic render start/completion times. It does not fabricate one merged
  detector batch.
- `~/status` is `CombinedOverlayStatus`, emitted every 200 ms with a one-second
  expiry. It reports `idle` without demand, `waiting_for_pair` without a recent
  complete pair, `streaming`, or the most recent rendering `error`. Received
  counters are ordered frame/first model/second model. Matched, rendered,
  rejected, expired, evicted, replaced, superseded and failure counts expose
  missing input, overload and rendering progress independently of inference.

The first model is amber and the second cyan, including invalid-depth boxes.
Each has its own model/count/truncation legend; `2D only` still marks unavailable
depth when `overlay_spatial=true`. The same spatial and capture-marker parameters
are supported and all parameters are startup-only. The node uses the physical
host's monotonic boot clock; simulated or remote clock domains are unsupported.

## Optional display measurement marker

`overlay_capture_marker=true` burns a binary acquisition marker into the same
image before encoding. It is intended for bounded integration measurements, not
browser-side box association. Images must be at least 496 × 32 pixels. There are
120 cells of 8 × 8 pixels, arranged in two rows of 60 starting at `(8, height-24)`.
Bits are least-significant-first within each byte. The 15 bytes are:

1. `0xA5` synchronization byte;
2. original ROS acquisition timestamp, unsigned 64-bit little endian;
3. capture sequence modulo 2³², unsigned 32-bit little endian;
4. CRC-16/CCITT, initial `0xffff`, polynomial `0x1021`, over the first 13 bytes,
   stored little endian.

The marker can establish which acquisition a decoded frame represents. Display
latency still needs a receiver presentation measurement and a measured clock
offset/uncertainty. SDK timestamp mapping uncertainty is a separate limitation.

## Verification

`detector_adapter` checks the retained image/result association across delayed
inference, invalid spatial estimates, and a subsequent empty result.
`detector_overlay` checks source immutability, image metadata, marker payload,
latest-only retention, supersession, expiry, mismatch rejection, and progress
while a real ROS subscriber executor is paused. These are synthetic captures;
they do not establish hardware or remote display performance.

`combined_overlay` adds independent model colours, valid empty results, exact
pairing/provenance, shuffled stream arrival, missing-partner bounds, no-consumer
release, generation invalidation and expiry during a held renderer. Its actual
ROS callbacks continue accepting both result streams while rendering is held.
