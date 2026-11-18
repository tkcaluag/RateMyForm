# Results and Outcomes: Tracy's section (CPSC 490 midterm poster)

Everything below comes from code that was run on 2026-11-17. Anything simulated is
labeled **SYNTHETIC**. Anything that needs real footage is labeled **NEEDS FOOTAGE**.

## 0. Read this first: what you can claim today

| You CAN say | You should NOT say (yet) |
|---|---|
| "We built a working prototype that runs locally: upload, measurements, pose detection, rep counting, scoring, personalized feedback." | "It is X% accurate." (No real-video accuracy exists yet.) |
| "On 16 synthetic test scenarios with known answers, it counted all 76 reps correctly, including noise, missing frames, and 24-120 fps." | "We tested it on participants." (Unless the team ran `evaluate_videos.py` on real recordings.) |
| "166 automated tests pass." | "It tells you if your squat is safe." |
| "Personalization changes the result: the same 50° lean scores 10.0 for a long-femur body and 7.6 for a short-femur body (synthetic)." | "Our thresholds are research-validated." (They're labeled PROTOTYPE until Billy's research replaces them.) |

**Action item for the team:** the poster's Methodology says "We recorded lifts with
different camera angles, distances, lighting, and clothing." If those recordings
exist, run `scripts/evaluate_videos.py` on them and add the real numbers. If they
don't, change that bullet to "We will test with…".

## 1. Poster box text (drop-in, about 120 words)

> **Results and Outcomes**
> - **Working prototype:** Streamlit app → upload a side-view squat video + 4 body
>   measurements → MediaPipe tracks 33 landmarks → reps counted → each rep scored /10
>   with explained deductions.
> - **Rep detection:** a two-threshold state machine (140°/160°) counted **76/76 reps
>   across 16 synthetic test scenarios**, including noise, dropouts, occlusion,
>   partial reps, and 24-120 fps *(synthetic data with known ground truth)*.
> - **Personalization works in both directions:** the same 50° torso lean scored
>   **9.0 → 10.0** for a long-femur body and **9.0 → 7.6** for a short-femur body
>   *(synthetic)*.
> - **Reliability:** hidden joints are reported as "not measured," never as bad form.
> - **166 automated tests pass.** Real-video accuracy testing is next.

## 2. Visuals (files ready in this repo)

| # | Requested evidence | File | Label on poster |
|---|---|---|---|
| 1 | App screenshots | `docs/presentation/screenshots/01_stage1_exercise_selection.png`, `02_stage2_upload_and_measurements.png`, `04_results_summary_SYNTHETIC.png` | "Prototype UI" |
| 2 | Pose detection screenshot | `results/charts/skeleton_keyframe_synthetic.png` (synthetic stick figure). **NEEDS FOOTAGE** for a real one: run `scripts/analyze_video.py` on a squat video → `rep_1_bottom.png` | "SYNTHETIC skeleton" until replaced |
| 3 | Real calculated joint angles | Synthetic rep table in `04_results_summary_SYNTHETIC.png` (knee 80°, hip 60.6°, torso 50°, shin 30°). **NEEDS FOOTAGE** for real angles: `reps.csv` from `analyze_video.py` | "SYNTHETIC" |
| 4 | Rep detection demo | `results/charts/knee_timeline_long_femur_synthetic.png` and `noise_smoothing_synthetic.png` | "SYNTHETIC" |
| 5 | Scoring breakdown | table in §3 below; screenshot `07_results_rep_detail_SYNTHETIC.png` | |
| 6 | Baseline vs personalized | `results/charts/baseline_vs_personalized_synthetic.png`, `personalization_curve.png`, `06_results_personalization_SYNTHETIC.png` | |
| 7 | Chart from executed tests | `results/charts/rep_detection_synthetic.png` (16/16 scenarios exact) | "SYNTHETIC test scenarios" |

The screenshots are 800 px wide. For sharper poster
images, run the app and take full-screen screenshots yourself (Win+Shift+S) on the
same pages.

**Best three for a small poster box:** (a) `baseline_vs_personalized_synthetic.png`,
(b) `knee_timeline_long_femur_synthetic.png`, (c) the results-page screenshot. These
three show what's original about the project, how it works, and that it's real.

## 3. Scoring breakdown example (actual app output, SYNTHETIC input)

Rep 2 of the "shallow reps" scenario:

| Step | Value |
|---|---|
| Start | 10.0 |
| Knee angle at bottom | 120° (reference: ≤ 100°, the poster's tip) |
| Deduction rule | 0.1 point per degree over the reference, max 3 → 20° × 0.1 |
| Deduction | **−2.0** |
| Torso lean 35° vs. limit 45° | no deduction |
| Chest drop 0° vs. 10° | no deduction |
| **Rep score** | **8.0 / 10** |

App message, verbatim: *"Rep 2: your knee angle at the bottom was about 120 deg,
above the 100 deg reference depth (-2.0)."*

## 4. Baseline vs. personalized example (SYNTHETIC)

Same movement (50° torso lean at the bottom, knee 80°), different bodies:

| Body | femur/torso | Torso-lean limit | Baseline score | Personalized score |
|---|---|---|---|---|
| Long femur (torso 48, femur 50, shin 42 cm) | 1.04 | 45° → **55°** (+12° suggested, capped at +10°) | 9.0 | **10.0** |
| Short femur (torso 55, femur 38, shin 42 cm) | 0.69 | 45° → **37.9°** (−7.1°) | 9.0 | **7.6** |

Baseline message: *"your torso leaned about 50 deg from vertical at the bottom,
farther forward than the 45 deg reference (-1.0)."* With personalization, the
long-femur user gets no deduction, because their proportions predict more forward
lean.

## 5. Implemented features (concise list)

1. Three-screen Streamlit app: exercise selection → upload + measurements → results
2. Input validation: file type, size, length, fps, and resolution; cm/in with range
   checks
3. MediaPipe Pose (Tasks API, full model) + OpenCV, 33 landmarks per frame, local
   only
4. Knee and hip angles (3D), torso and shin lean (2D), depth classification
5. Smoothing, visibility checks, side-view check
6. Rep counter: 140°/160° hysteresis, debounce, partial/incomplete rep reporting
7. Explainable 0-10 scoring with per-rep deductions
8. Baseline vs. personalized evaluation using body ratios (bounded)
9. Measurement check: entered ratios vs. ratios estimated from video
10. Skeleton overlay video, rep key frames, charts, JSON/CSV export
11. 166 automated tests + a repeatable evaluation script

## 6. Limitations (say these out loud; it builds trust)

- No real-video accuracy yet; synthetic tests prove the logic, not real-world accuracy.
- Single camera: depth is estimated, so lean and depth need a true side view.
- Reference values are partly prototype placeholders until guideline research is done.
- Personalization is a simple balance model, bounded to ±10°, and not yet validated.
- Squats only; one side of the body; no knee-cave detection (needs a front view).

## 7. Speaking notes (about 90 seconds)

> "For results, the main outcome is that we have a working prototype, not just a
> plan. You open the app, pick squats, upload a side-view video, and type in four
> measurements. MediaPipe finds 33 points on your body in every frame, and we turn
> those into knee angle, hip angle, torso lean, shin lean, and depth.
>
> To count reps we use two thresholds. A rep starts when your knee bends past 140
> degrees, and it only finishes when you straighten past 160. That gap means small
> jitter in the tracking can't double-count a rep. We tested this on 16 simulated
> squat sets where we already knew the right answer, with noise, missing frames,
> half reps, and different frame rates, and it counted all 76 reps correctly. I want
> to be clear that those are simulated tests. They show the logic works. Testing on
> real recordings is our next step.
>
> Each rep starts at 10 and loses points for specific things, like not reaching the
> depth target, and the app tells you exactly why. The part that makes our project
> different is personalization. This chart is the same 50-degree lean scored for two
> different bodies. Someone with long thighs gets a 10, because their proportions
> need more lean to stay balanced. Someone with short thighs gets a 7.6. So
> personalization isn't just being nicer; it adjusts in both directions, and it's
> capped so one bad measurement can't swing the score.
>
> Finally, if the camera can't see a joint, we say 'not measured' instead of
> marking your form as bad. This is a prototype score, not medical advice."

## 8. Likely professor questions (short answers)

**Q: How accurate is it?**
A: We haven't measured real-world accuracy yet. On synthetic data with known
answers, rep counting was 76/76 and angle error was about 2-3° with added noise.
Next, we'll compare against manually counted reps on recorded videos.

**Q: Why two thresholds instead of one?**
A: With one threshold, jitter around 140° could count several reps. The 140/160 gap
is hysteresis: the angle has to clearly go down and clearly come back up. We also
require the crossing to last 0.1 s, and a rep has to take at least 0.6 s.

**Q: How do body measurements change the feedback?**
A: We compute femur/torso and lower-leg/femur. A simple balance model says longer
femurs need more forward lean at parallel. We shift the torso-lean limit by the
difference between your prediction and a reference body, capped at ±10°.

**Q: Is that personalization scientifically validated?**
A: Not yet. Research supports the direction: when the knees can't travel forward,
load shifts to the hips (Fry et al., 2003), and Lorenzetti et al. (2009) describe the
trunk bending forward to compensate. They also found a much smaller hip effect than
Fry. Our specific numbers are a prototype heuristic, which is why the app shows both
the baseline and the personalized scores.

**Q: What if the camera angle is wrong or a joint is hidden?**
A: We check whether it's a side view. If it isn't, we don't score lean or depth.
Hidden joints are marked "not measured," and a rep with nothing reliable gets no
score instead of a fake 10.

**Q: Why MediaPipe? Why 3D landmarks?**
A: It's free, runs on a CPU, and gives 33 landmarks plus 3D world coordinates.
Angles between body parts in 3D don't depend on camera position. Lean needs
"vertical," so we take that from the 2D image.

**Q: What about privacy?**
A: Everything runs on the user's computer. The video is deleted after analysis,
nothing is uploaded, and the app only listens on localhost. Reports don't include
the video.

**Q: Did you use AI to generate feedback?**
A: No. Feedback is rules-based, so every sentence traces back to a measured number.
We compared LLM feedback as a future option; the risk is advice that doesn't match
what the video shows.

**Q: What happened to the 75% accuracy ML goal?**
A: It's still planned. First we need labeled data and a clear definition of a
"correct" form ranking.
