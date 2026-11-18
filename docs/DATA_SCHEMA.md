# Input / output data schema

## Inputs

### Body measurements (UI, CLI, or `BodyMeasurements`)

| Field | Unit entered | Stored as | Accepted range (cm) | Used for |
|---|---|---|---|---|
| torso | cm or in | `torso_cm` | 25-90 | femur/torso ratio |
| femur | cm or in | `femur_cm` | 20-75 | femur/torso, lower-leg/femur |
| lower_leg | cm or in | `lower_leg_cm` | 20-75 | lower-leg/femur |
| arm | cm or in | `arm_cm` | 35-110 | arm/torso check only (not scored) |

Inches are converted with 2.54 cm/in. The ranges are input-error guards, not
anthropometric norms. A ratio outside femur/torso 0.5-1.5 or lower-leg/femur 0.6-1.5
produces a warning, not an error.

### Video

| Check | Limit | Where |
|---|---|---|
| Extension | .mp4 .mov .avi .m4v | `validate_upload` |
| Size | ≤ 200 MB, > 0 | `validate_upload` + Streamlit `maxUploadSize` |
| Duration | 1.5-60 s | `validate_video_properties` |
| Frame rate | 10-240 fps (analyzed at ≤ 30 fps) | `validate_video_properties`, `frame_stride` |
| Resolution | short side ≥ 240 px (downscaled to ≤ 960 px long side) | same |

The upload is saved as `upload.<ext>` in a new temporary folder, never under the
user's filename, and the folder is deleted after analysis.

## Internal: `LandmarkSequence` (`ratemyform/landmarks.py`)

| Array | Shape | Meaning |
|---|---|---|
| `timestamps_s` | (N,) | seconds; strictly increasing |
| `frame_indices` | (N,) | source frame index (after fps subsampling) |
| `detected` | (N,) bool | MediaPipe found a person |
| `image_xy` | (N, 33, 2) | pixel coordinates; NaN if not detected |
| `visibility` | (N, 33) | 0-1, probability in frame and not occluded |
| `world_xyz` | (N, 33, 3) | meters, origin = hip midpoint; NaN if not detected |
| `width`, `height`, `fps`, `source`, `metadata` | scalars | `source` is `"video"` or `"synthetic"` |

Any pose source that fills these arrays can feed `analyze_sequence()`.

## Output: JSON report (`AnalysisResult.to_dict()`, schema 1.0)

Download it from the Export tab or `scripts/analyze_video.py` (`report.json`). Full
examples: [results/example_report_forward_lean_long_femur.json](../results/example_report_forward_lean_long_femur.json).

```text
schema_version   "1.0"
exercise         "squat"
source           "video" | "synthetic"
disclaimer       prototype / non-clinical statement
input
  video          {fps, frame_count, width, height, duration_s}     (empty for synthetic)
  measurements_cm {torso_cm, femur_cm, lower_leg_cm, arm_cm}
  entered_ratios {femur_to_torso, lower_leg_to_femur, arm_to_torso}
processing       {frames_analyzed, frames_with_person, detection_rate, analysis_fps,
                  joint_angle_source, facing, side_visibility{left,right},
                  median_knee_2d_vs_3d_difference_deg, pose_seconds, analysis_seconds,
                  overlay_seconds, total_seconds, model_variant}
camera
  side_analyzed  "left" | "right"
  orientation    {label: side|angled|front_or_back|unknown, index, message}
ratio_checks[]   {name, entered, video_estimate, relative_difference, consistent, message}
video_ratio_estimates {femur_to_torso, lower_leg_to_femur, arm_to_torso, frames_used}
thresholds
  baseline       {label, depth_target_knee_angle_deg, torso_lean_limit_deg,
                  chest_drop_threshold_deg, torso_lean_adjustment_deg, rationale[]}
  personalized   (same fields)
summary          {rep_count, overall_score_baseline, overall_score_personalized,
                  feedback[], warnings[]}
reps[]
  measurements   {rep_number, start_time_s, bottom_time_s, end_time_s, duration_s,
                  knee_angle_deg, hip_angle_deg, torso_lean_deg, shin_lean_deg,
                  depth_ratio, depth_class, chest_drop_deg,
                  knee_angle_2d_deg, knee_angle_3d_deg,
                  reliable{knee_angle, hip_angle, torso_lean, shin_lean, depth_ratio, chest_drop},
                  reliability_notes[], low_confidence}
  baseline       {rep_number, label, score|null, deductions[], not_scored[]}
  personalized   (same)
    deductions[] {rule: depth|torso_lean|chest_drop, points, measured_value,
                  reference_value, unit, message, personalized_reference}
baseline_vs_personalized[] {rep_number, baseline_score, personalized_score,
                            difference, rules_changed[]}
uncounted_movements[] {kind: partial|incomplete|too_fast|too_slow|started_mid_rep,
                       start_time_s, end_time_s, min_knee_angle_deg, reason}
```

Conventions:

- Angles are in degrees, rounded to 0.1. A score of `null` means no rule could be
  scored reliably for that rep; it is **not** a 0 or a 10.
- `overall_score_*` is the mean of the non-null rep scores, or `null`.
- The report never contains the video, images, or the uploaded filename.

## Output: CSV exports

`ratemyform_reps.csv` (one row per rep): `rep, bottom_time_s, knee_angle_deg,
hip_angle_deg, torso_lean_deg, shin_lean_deg, depth, chest_drop_deg, baseline_score,
personalized_score, low_confidence`

`ratemyform_timeseries.csv` (one row per analyzed frame): `time_s, state,
knee_angle, hip_angle, torso_lean, shin_lean, depth_ratio` (smoothed; empty means
not measured)

## Evaluation outputs (`results/`)

| File | Content |
|---|---|
| `synthetic_evaluation.csv` / `.json` | one row per SYNTHETIC scenario: expected vs. detected reps, angle errors, scores, timing; JSON adds the environment |
| `charts/*.png` | charts generated from those runs |
| `real/video_evaluation.csv` / `.json` | produced by `evaluate_videos.py` from real recordings (git-ignored) |
| `data/synthetic/*_landmarks.csv` | sample SYNTHETIC per-frame landmarks + ground-truth angles |
