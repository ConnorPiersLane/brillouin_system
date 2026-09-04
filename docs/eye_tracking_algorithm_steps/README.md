# Eye-tracking pupil detection — algorithm steps

One PNG per stage of the production 2D pupil-detection pipeline
(`src/brillouin_system/eye_tracker/pupil_fitting/ellipse_fitter_helpers.py`,
`find_pupil_ellipse_with_flooding` plus the circular pre-mask from
`EllipseFitter`). Generated from `tests/eye_tracking/left/connor.png` with
parameters tuned for that image (threshold 25, mask radius 130 px, mask
center offset (91, -6); live values come from
`eye_tracker_config/eye_tracker_config.toml`).

| Image | Step | Code |
|---|---|---|
| `step1_original.png` | Raw grayscale camera frame (normalized to uint8) | `_ensure_u8` |
| `step2_circular_mask.png` | Black out everything outside a circle around the expected pupil position | `make_img_black_outside_ring_around_center` |
| `step3_binary_threshold.png` | Inverse binary threshold — dark pixels (pupil, masked area) become white | `cv2.threshold(..., THRESH_BINARY_INV)` |
| `step4_floodfill_background_removed.png` | Flood fill from the corner removes all white connected to the border; only isolated dark blobs (the pupil) survive | `cv2.floodFill` |
| `step5_fill_vertical_gaps.png` | Bridge small vertical gaps (eyelashes crossing the pupil) | `fill_vertical_gaps_binary_fast` |
| `step6_largest_component.png` | Keep only the largest connected component | `cv2.connectedComponentsWithStats` |
| `step7_ellipse_fit.png` | Fit an ellipse to the component contour; overlay on the original (green = ellipse, red = center) | `cv2.fitEllipse` |

Downstream (not image-based): the left/right ellipses are lifted to 3D view
cones and intersected to get the 3D pupil center, normal, and radius
(`pupil_fitting/pupil_detector.py`, `triangulate_center_using_cones`).

Fitted result for this image: center (595.7, 532.0) px, axes (70.2, 83.0) px,
angle 6.8°.
