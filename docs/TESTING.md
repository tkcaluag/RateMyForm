# Testing and evaluation

Environment for every result below: Windows 11 Home (10.0.26300), AMD Ryzen 9 9900X,
61.6 GB RAM, CPU only, Python 3.13.17, mediapipe 1.2.3, opencv-contrib-python
4.12.0.88, numpy 2.2.6, streamlit 1.67.0. Date: 2026-11-17.

## How to run

```bash
.venv\Scripts\python.exe -m pytest
```

```bash
.venv\Scripts\python.exe scripts\run_evaluation.py
```

Optional subsets: `-m "not integration and not ui"` (pure logic, no model needed),
`-m integration` (real MediaPipe + OpenCV), `-m ui` (Streamlit AppTest).

## Automated test results (executed)

`pytest`: **166 passed, 0 failed, 0 skipped** (about 6 s).

| File | Tests | What it checks |
|---|---|---|
| `test_geometry.py` | 11 | angle formula (2D/3D, 45/90/180 deg), symmetry, NaN/degenerate input, signed lean vs. facing |
| `test_smoothing.py` | 6 | odd windows scale with fps; short gaps filled, long and edge gaps not; spike removal |
| `test_rep_counter.py` | 22 | clean reps; jitter in the hysteresis band counted once; noise near 160 deg ignored; one-frame spike debounced; too-fast, partial, incomplete, and started-mid-rep movements; tracking gap → low confidence; 15-240 fps independence; streaming == batch; config validation |
| `test_squat_metrics.py` | 12 | angles match ground truth; 2D == 3D in a side view; side and facing detection; missing landmarks → NaN; occlusion only invalidates dependent metrics; side/angled/front orientation; video ratio estimates; noisy-input error bound |
| `test_personalization.py` | 10 | ratio math; model direction (longer femur → more lean, longer shin → less); hand-calculated value; zero adjustment at reference; ±10 clamp; only torso lean personalized; rationale labeled as heuristic; ratio mismatch flags |
| `test_scoring.py` | 12 | exact deduction math and caps; unreliable metric → not deducted; nothing reliable → `null`, not 10; overall ignores unscored reps; determinism; long-femur personalized vs. baseline |
| `test_validation.py` | 27 | cm/in conversion; missing, non-numeric, boolean, negative, NaN, inf, and out-of-range values; ratio warnings; extensions; size limits; fps/duration/resolution; path-traversal-safe temp names |
| `test_pipeline_synthetic.py` | 47 | every SYNTHETIC scenario: rep count == ground truth, JSON-serializable report; depth, lean, and chest-drop behavior; hidden shoulder not penalized; tracking gap unscored; front view flagged; fps invariance; identical inputs → identical report; no-person video; measurement-typo warning |
| `test_pose_detection.py` | 14 | **real MediaPipe + OpenCV:** landmark indices and 35 skeleton connections equal MediaPipe's own enums; fps stride; probe + corrupt-file error; blank image → no detection; VIDEO mode with 60 → 30 fps subsampling and increasing timestamps; person-free video → safe "no score" result, valid WebM overlay, temp files removed |
| `test_app.py` | 5 | Streamlit AppTest: stage 1 content; Analyze disabled until valid; out-of-range error shown; cm→in conversion; synthetic demo reaches results with SYNTHETIC banner and expected metrics; Start over |

### Failures found and fixed during development (root causes, not test edits)

1. **Debounce floating-point boundary.** At 30 fps, 3 frame steps (0.1 s) computed
   as 0.0999…, so a 0.1 s debounce sometimes rejected a real crossing, depending on
   frame-rate rounding. Fixed with a 1 µs tolerance in `RepCounter._held_long_enough`.
   The test kept its assertion.
2. **Inconsistent synthetic geometry.** The generator put hips wider than knees, so
   its simulated 3D leg was not planar and its 3D knee angle (176 deg standing)
   disagreed with its own 180 deg ground truth. Fixed in the generator.
3. One test expectation (`test_video_ratio_estimates_match_skeleton`) assumed a flat
   torso. With shoulders wider than hips, the 3D shoulder-hip distance is longer
   than the side-view length. The expected value now uses the real 3D geometry at
   the same tight tolerance; the comment in the test explains it.

## Synthetic evaluation (executed, SYNTHETIC data)

`scripts/run_evaluation.py` → `results/synthetic_evaluation.csv`, `.json`, `charts/`.

- **Rep counting: 16 of 16 scenarios exact (76 of 76 reps)**, including noise and
  spikes, a bounce across 140 deg, 15% random dropouts, occlusion, a 0.8 s tracking
  gap, an incomplete last rep, two partial dips, a front view, and 24/30/60/120 fps.
- Knee-angle error on the noisy scenario (4 px noise + 5% spike frames): 2.85 deg
  mean per frame after smoothing, and 1.62 deg at rep bottoms.
- Bottom-bounce scenario: the bottom knee angle reads 3.5 deg high, because smoothing
  flattens the lowest oscillations. This is a known trade-off of smoothing.
- Front-view scenario: torso-lean error is 24 deg, which is exactly why the app
  refuses to score lean/depth when the orientation check fails.
- Personalization: long femur 9.0 → 10.0, short femur 9.0 → 7.6, and unchanged where
  lean was not the issue (shallow reps 8.2 → 8.2).

**What this proves:** the logic does what it is designed to do on controlled inputs
with known answers. **What it does not prove:** accuracy on real people, real cameras,
real clothing, or real lighting. Do not present these as real-world accuracy.

## Performance

| Measurement | Result | Conditions |
|---|---|---|
| Analysis after pose detection (metrics + reps + scoring) | median ≈ 30 ms per 18 s set (532 frames, 30 fps); 123 ms at 120 fps | SYNTHETIC landmarks, median of 5 runs |
| Pose detection, `pose_landmarker_full` | 3.15 s / 300 frames ≈ **10.5 ms per frame** | 1080×1920 generated video with **no person**, downscaled to 540×960. Person-free frames take a different path inside MediaPipe, so this is not representative of real footage. |
| Overlay rendering + VP8 encoding | 3.05 s / 300 frames ≈ 10 ms per frame | same video |
| End-to-end CLI on that video | 6.2 s total | same video |

Real-footage throughput, memory, and Streamlit latency have **not been measured**.
`evaluate_videos.py` records `ms_per_frame` for each real video.

## Manual / live checks (executed)

- Started the app (`streamlit run app.py`) and walked stage 1 → stage 2 → results in
  the browser at 1366×860. Out-of-range femur showed the error and kept Analyze
  disabled; valid values showed ratios; the synthetic demo rendered scores, the
  rep table, charts (no label collisions after fixes), the personalization
  explanation, and rep key frames. No browser console errors.
- The server log confirmed the app listens on `localhost` only after the config change.
- The `scripts/analyze_video.py` CLI ran end to end and wrote `report.json`, CSVs,
  `overlay.webm`, and charts.

## NOT TESTED (needs action)

| Item | Why | How to test |
|---|---|---|
| Pose detection on real people; rep-count accuracy vs. manual counts | No licensed or consented real footage available | Record per [data/real_videos/README.md](../data/real_videos/README.md), run `evaluate_videos.py` |
| Camera angle, distance, lighting, and clothing sensitivity on real video | same | same protocol (manifest columns) |
| Browser file-upload widget with a real file | AppTest cannot drive uploads | Upload a phone video manually |
| iPhone .MOV (HEVC), rotation metadata, variable frame rate | No sample files | Record on iPhone and Android, check rep timing |
| Safari, Firefox, mobile browsers | Only a Chromium-based browser was used | Open http://localhost:8501 in each |
| macOS / Linux install | Only Windows 11 was used | Follow the README on those systems |
| Accessibility audit (screen reader, contrast of Streamlit theme) | Not performed | Keyboard-only pass + a screen reader |
| Long videos near the 60 s / 200 MB limits | Not run | Time a 60 s 1080p clip |
