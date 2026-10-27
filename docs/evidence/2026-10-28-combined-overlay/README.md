# Separate combined presentation component — 28 October 2026

The four native checks passed on reserved Jah using ROS Jazzy and an isolated
source/build/install workspace: `detector_contract`, `detector_adapter`,
`detector_overlay` and `combined_overlay`. The tested implementation is
`db77e7e`; [software.json](software.json) records full revisions, log hashes and
installed artifact identity. The existing TensorRT FP16 deprecation warning
remains visible; no engine rebuild or inference change was needed.

The new synthetic cases use actual ROS subscriptions with an injected renderer.
They check distinct model colours, immutable padded source images, both original
batches, an empty model result, shuffled arrival between streams, full capture
and depth-pairing mismatch rejection, missing-partner bounds, no-consumer release,
one replaceable pending render, input callback progress during a held render,
generation invalidation, incomplete-join expiry and the stricter model age limit
for both output and status. Existing single-model renderer/adapter checks passed.

The final ROS install preserves the deployment's isolated package layout.
284 native files were verified against the build output; project libraries
resolve from the candidate release. The runtime's 100 package files remain
byte-identical to dev21, so its prior unit/browser fixture suites were not rerun.

The sibling `robot_composition` repository records separate live D455/OAK-D
evidence: 50 exact combined images, comparison against both independently
received batches, and uninterrupted detector progress during viewer selection.
Synthetic checks alone do not establish hardware accuracy or display performance.
