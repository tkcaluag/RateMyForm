# Limitations and assumptions

## What the scores are (and are not)

- Prototype estimates from one camera. They are **not** a validated injury-risk,
  medical, or clinical assessment, and they do not replace a certified trainer,
  physical therapist, or medical professional.
- No real-world accuracy has been measured yet. All accuracy-style numbers in this
  repository come from SYNTHETIC data.
- The reference values are partly from our methodology (140/160 deg rep thresholds,
  100 deg depth tip, start at 10) and partly PROTOTYPE choices (45 deg lean limit,
  deduction rates, chest-drop rule). They are not research-validated guidelines yet.

## Computer-vision limitations

- **Monocular pose estimation.** MediaPipe estimates joint positions from a 2D image.
  The model card says depth (z) comes from synthetic training data and is "not metric
  but up to scale." World coordinates are estimates in meters, not measurements.
- **Joint centers ≠ anatomical landmarks.** The hip landmark is the hip joint, not
  the hip crease used for powerlifting depth, so "parallel" is approximate.
- **Camera setup matters.** Lean and depth assume a side view with an upright, level
  phone. The app detects front or back views and refuses to score lean/depth there,
  but it cannot detect a tilted phone.
- **Model card out-of-scope cases:** multiple people in view (only one is tracked),
  people farther than about 4 m, head not visible, and applications needing
  metric-accurate depth.
- **Conditions that degrade tracking** (proposal and model card): low or uneven
  light, loose clothing, motion blur, occlusion by equipment, and plates or a bar
  hiding the hips. Missing landmarks are reported as "not measured."
- **Frame timing** assumes a constant frame rate (frame index ÷ fps).
  Variable-frame-rate phone videos may have small timing errors. This is untested.
- **Single side.** Only the side facing the camera is analyzed. Left/right asymmetry
  and knee valgus (knees caving in) need a front view and are not evaluated.

## Analysis limitations

- **Fixed thresholds.** A rep needs the knee below 140 deg and back above 160 deg.
  Someone whose standing knee angle reads under 160 deg (camera angle, clothing) gets
  no reps counted. The app warns when standing is never detected.
- **Smoothing trade-off.** Smoothing removes jitter but can raise the measured bottom
  angle for very short or bouncing bottoms (3.5 deg in the synthetic bounce test).
- **Bottom = smallest knee angle.** Other metrics are read at that frame, not at
  their own extremes.
- **Personalization is a heuristic.** It is a geometric balance model: bounded,
  relative, and only for torso lean. It ignores mobility, flexibility, injuries,
  muscle mass, bar position, and footwear. See [PERSONALIZATION.md](PERSONALIZATION.md).
- **Arm length** is collected (per the proposal) and checked against the video, but
  it is not used in squat scoring.
- **One exercise.** Squats only; others are shown as planned.

## Product and engineering limitations

- Runs locally for one user at a time; there are no accounts or history.
- The overlay video is WebM (VP8). Safari playback is untested.
- On the development laptop, newer OpenCV and pyarrow wheels were blocked by Windows
  Application Control, so older versions are pinned.
- The app was tested in a Chromium-based browser on Windows only.

## Ethical and privacy assumptions (Project Considerations)

- Users should record where they will not capture bystanders.
- Feedback reports measurable information ("reference range," "about 52 deg"),
  never "safe/unsafe," and uses no personal characteristics beyond the entered
  lengths.
- The model card reports fairness evaluation across regions, skin tones, and gender
  (for example, the full model's skin-tone range is 85.9-92.9% PDJ). Our app inherits
  any such differences, and we have not evaluated them ourselves.
