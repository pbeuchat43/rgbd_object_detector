# Exact-capture overlay verification — 27 October 2026

The native `detector_contract`, `detector_adapter` and `detector_overlay` CTest
executables passed on reserved Jah, with no failed or skipped tests. The retained
[log](native-tests.log) includes an intentional synthetic backend error that the
adapter test verifies; it is not a hardware inference failure.

Checks cover immutable input pixels, capture/header equality, matching delayed
inference against two different images, clearing boxes on empty results,
invalid-spatial 2D boxes, the optional capture marker, replacement bounds,
in-flight generation invalidation, stale results, no-consumer gating, renderer
failure isolation and a real ROS subscriber whose executor stops consuming.
The first renderer fixture incorrectly inspected a box corner covered by its
own label; the corrected check inspects the unobscured side edge. Rendering
behavior was unchanged by that fixture correction.

The composed adapter and worker keep inference independent of presentation.
A downstream standalone CMake consumer subsequently built and linked the public
overlay library after correcting the exported targets/dependencies. The tested
integration implementation is `01e36cb`; that export correction did not change
renderer, decoder or worker source. The sibling `robot_composition` repository
records 250 exact original-image/batch/overlay comparisons across ten live
camera/model/rate cases, plus separate relocated/persistent deployment checks.

These checks do not establish moving-scene quality, physical display scanout,
metric spatial accuracy or physical USB recovery. Private camera imagery is not
included. Model reference evidence is retained in
[the selectable-model record](../2026-10-27-models/README.md).
