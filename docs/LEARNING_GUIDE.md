# RateMyForm learning guide

One concept per section. Read the section, look at the code it points to, then
answer the quiz before opening the answer.

---

## 1. The big picture: a pipeline of small steps

```
video ─► frames ─► 33 landmarks ─► angles ─► smooth ─► count reps ─► measure bottoms ─► score ─► feedback
        (OpenCV)   (MediaPipe)     (math)    (NumPy)   (state machine)                  (rules)
```

Each arrow is a module in `ratemyform/`. Each step's output is the next step's
input, which is why we can test each piece alone. The synthetic generator skips
the first two steps and produces landmarks directly.

**Quiz 1.** Why can we test the rep counter without any video file?

<details><summary>Answer</summary>The rep counter only needs a list of times and
knee angles. Tests build those lists by hand (or from the synthetic generator), so
no video or MediaPipe is involved.</details>

---

## 2. What MediaPipe gives us

MediaPipe Pose looks at one frame and returns **33 landmarks**: nose, shoulders,
elbows, wrists, hips, knees, ankles, heels, toes, and more. Each landmark comes in two
versions:

| | `pose_landmarks` (image) | `pose_world_landmarks` (world) |
|---|---|---|
| Units | fractions of width/height (0-1) | meters |
| Origin | top-left of the image | middle of the hips |
| Good for | drawing; anything relative to "up" in the picture | angles between body parts |

Each landmark also has **visibility**: the model's probability (0-1) that the point is
in frame and not hidden. We ignore landmarks below 0.5.

Code: `pose_detection.py::_result_to_arrays` multiplies x by width and y by height,
which converts fractions to pixels.

**Quiz 2.** Why do we convert normalized coordinates to pixels before measuring an
angle on a 1080×1920 phone video?

<details><summary>Answer</summary>Normalized x is divided by 1080 and y by 1920, so
the axes are squished by different amounts. Angles measured in those units would be
distorted. Pixels have the same scale on both axes.</details>

---

## 3. Joint angles with the dot product

Knee angle = the angle at the knee between the knee→hip and knee→ankle vectors.

```
cos(θ) = (BA · BC) / (|BA| |BC|)        B = knee, A = hip, C = ankle
```

Worked example (2D pixels): hip (0, 0), knee (0, 100), ankle (100, 100).
BA = (0, −100), BC = (100, 0), BA·BC = 0 → cos θ = 0 → **θ = 90°**.

A straight leg gives 180°, and a deep squat gives a small angle.

Code: `geometry.py::angle_at`. It returns NaN if a point is missing or two points
coincide, instead of crashing.

**Quiz 3.** Hip (0, 0), knee (0, 1), ankle (0, 2). What is the knee angle, and what
position is that?

<details><summary>Answer</summary>BA = (0, −1), BC = (0, 1), dot = −1 → cos θ = −1 →
θ = 180°. A straight leg (standing).</details>

---

## 4. Lean from vertical

Torso lean = how far the hip→shoulder line tips away from straight up. In images, y
grows **downward**, so "up" is (0, −1). We use `atan2(dx, −dy)`, which gives 0° for
vertical and a positive value when the person leans the way they face.

The facing direction comes from toes vs. heel. Code:
`squat_metrics.py::estimate_facing`.

**Quiz 4.** Why do we measure lean in the 2D image and not with the 3D world
landmarks?

<details><summary>Answer</summary>Lean is relative to gravity ("vertical"). The
image's vertical axis matches gravity if the phone is upright. MediaPipe's world
coordinates don't guarantee an axis aligned with gravity.</details>

---

## 5. Smoothing: median, then mean

Pose estimates jitter. We clean each angle series:

1. Fill gaps ≤ 0.25 s by drawing a straight line between the neighbors.
2. **Rolling median** (0.15 s window): a single crazy frame can't win a median.
3. **Rolling mean** (0.15 s): averages out the small wobble.

Windows are in **seconds**, so a 60 fps video uses 9 frames where a 30 fps video uses
5.

**Quiz 5.** Values `[170, 170, 90, 170, 170]` with a window of 5. What do the median
and the mean give for the middle frame?

<details><summary>Answer</summary>Median = 170 (spike removed). Mean = 154 (spike
pulls it down). That's why the median runs first.</details>

---

## 6. The rep counter (state machine with hysteresis)

```
           knee < 140° for 0.1 s                 knee > 160° for 0.1 s
STANDING ─────────────────────────► SQUATTING ─────────────────────────► STANDING
   ▲                                                                   (+1 rep)
WAITING (until we first see standing)
```

- **Hysteresis:** going down needs < 140°, and coming up needs > 160°. Jitter between
  140 and 160 can't flip the state.
- **Debounce:** a crossing must last 0.1 s.
- **Duration check:** a "rep" shorter than 0.6 s is jitter, not a squat.
- **Bottom:** the frame with the smallest knee angle inside the rep.
- Reported but **not counted:** partial dips (below 155° but never below 140°), an
  unfinished last rep, and a video that starts mid-squat.

Code: `rep_counter.py::RepCounter.update`.

**Quiz 6.** During one squat, the knee angle goes 175 → 135 → 145 → 133 → 150 → 138 →
170. How many reps are counted, and why?

<details><summary>Answer</summary>One. It went below 140 (SQUATTING) and bounced
between 133 and 150, but never above 160 until the end, so the state never returned
to STANDING in between.</details>

---

## 7. Personalization in one formula

```
sin(predicted lean) = (femur/torso) × (1 − (shin/femur) × sin(30°))
adjustment = predicted(user) − predicted(reference), clamped to ±10°
personalized limit = 45° + adjustment
```

Longer femur → more lean allowed. Longer shins → less. We use only the
*difference* from a reference body, and we cap it. Full derivation:
[PERSONALIZATION.md](PERSONALIZATION.md).

**Quiz 7.** Why clamp the adjustment to ±10°?

<details><summary>Answer</summary>So a typo or bad measurement (e.g., femur 70 cm)
can't change the evaluation dramatically. It also keeps the unvalidated model from
overriding the reference range.</details>

---

## 8. Scoring and reliability

Each rep starts at 10. Rules: depth (0.1 per degree over 100°, max 3), torso lean
(0.2 per degree over the limit, max 3), chest drop (0.1 per degree over 10°, max 2).

Before scoring, each metric gets a **reliability** check: were its landmarks visible
near the bottom? Was tracking lost for more than 0.5 s? Is it a side view?
Unreliable metrics are **skipped and explained**, never deducted. If nothing is
reliable, the rep score is `None`, not 10.

**Quiz 8.** A rep's shoulder is hidden at the bottom, and the true lean was 60°.
What happens to the torso-lean score?

<details><summary>Answer</summary>Nothing is deducted. Torso lean is marked "not
scored: unreliable" with a note, because a missing landmark is a data problem, not
evidence of bad form. (Tested in `test_hidden_shoulder_is_not_scored_as_bad_form`.)</details>

---

## 9. How the Streamlit frontend talks to the analysis

Streamlit re-runs `app.py` from top to bottom on every click. To remember things
between reruns, we store them in `st.session_state` (`stage`, `result`, the
measurement values). When you press Analyze, `app.py`:

1. writes the upload to a temp folder as `upload.mp4` (never your filename);
2. calls `analyze_video(path, measurements, progress=...)`;
3. stores the returned `AnalysisResult` and switches to stage 3.

Then the temp folder is deleted. The UI only *displays* the result.

**Quiz 9.** Why does the app save the upload as `upload.mp4` instead of the user's
filename?

<details><summary>Answer</summary>A filename like `../../evil.mp4` could write
outside the temp folder (path traversal). It also keeps personal names out of
logs. (Tested in `test_temp_path_ignores_user_filename`.)</details>

---

## 10. Testing and debugging

- `pytest` runs all 166 tests; `-m "not integration and not ui"` runs the fast ones.
- **Synthetic data** gives known answers ("5 reps, bottom at 80°"), so a test can
  check exact numbers.
- **Debugging a wrong rep count on a real video:** run `scripts/analyze_video.py`,
  open `timeseries.csv` and `knee_timeline.png`, and check whether the knee angle
  actually crossed 140 and 160. If standing reads 155°, the camera angle or clothing
  is probably distorting the knee.
- When a test fails: reproduce it, find the root cause, and fix the code, not the
  test. Example from this project: a 0.1 s debounce failed because
  0.1 computed as 0.09999…, and the fix was a tiny tolerance in the comparison.

**Quiz 10.** Our synthetic tests all pass. Can we say "RateMyForm counts reps with
100% accuracy"?

<details><summary>Answer</summary>No. Synthetic tests show the logic is correct on
controlled inputs. Real-world accuracy needs real recordings compared against
manually counted reps (`scripts/evaluate_videos.py`).</details>
