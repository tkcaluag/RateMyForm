# Phase 1 audit: documents vs. code

Audit date: 2026-11-17. Sources reviewed (copies in [original_submissions/](original_submissions/)):

| Source | What it contributes |
|---|---|
| Project Statement & References (Idea 1, Tracy) | Need, preliminary requirements, limitations (camera, lighting, clothing, occlusion; not a substitute for professionals) |
| Project Proposal (Sep 16, 2026) | Smartphone video, pose estimation, joint angles, torso/femur/lower-leg/arm measurements, research-based guidelines, explanations, overall score (e.g., 8/10), testing plan (angles, distances, lighting, clothing, participants) |
| Status Report 2 | Who did what, what is done vs. planned, challenges, SMART goals |
| CPSC 490 Midterm Poster (Google Drive, read 2026-11-17) | Methodology: Streamlit, MediaPipe 33 landmarks, OpenCV, 3D landmarks, smoothing, 140/160 deg rep counter, ratio-based personalization, video-vs-entered ratio check, start-at-10 deductions |
| Project Considerations (Tracy) | Privacy of bystanders, non-discrimination, no extra hardware, report measurable info only and never claim "safe/unsafe" |
| Presentation rubric | Evidence-backed conclusions, clear visuals, ability to answer questions |

## 1. Team source code

**No team source code was available.** The working folder was empty. A Google Drive
search for MediaPipe, Streamlit, or `.py` files found only documents, and the GitHub
CLI is not installed. Status Report 2 describes code that exists somewhere:

- Tristan: camera feed with landmarks (Oct 24)
- Eric: UI with an upload prompt (Oct 26)
- Kyle: importable pose-detection functions with an overlay video and a webcam demo
  (Oct 28), and an analysis file with knee/hip/torso/shin/depth, rep counting,
  per-rep lowest point, and a measurement-vs-video comparison (Nov 1)

So this prototype is a **fresh implementation of those described interfaces**. It
does not replace or claim credit for teammates' code.

## 2. Requirement-by-requirement status

Status labels: **VERIFIED** = automated or live test passed in this environment;
**PARTIAL** = works, with an important gap noted; **NOT TESTED** = cannot be verified
without real footage; **PLANNED** = not built.

| Requirement (source) | Status | Evidence / notes |
|---|---|---|
| Choose an exercise; squats only, others shown as planned (proposal, prompt) | VERIFIED | `app.py` stage 1; `tests/test_app.py` |
| Upload a smartphone video (proposal) | PARTIAL | Upload, size/type/duration/fps validation, and safe temp handling are implemented and unit-tested. The browser upload widget was not driven by an automated test; the same code path is covered through `analyze_video` integration tests. |
| Enter torso, femur, lower-leg, arm lengths with validation and units (proposal) | VERIFIED | `validation.py`; 27 tests in `tests/test_validation.py`; live UI check showed an out-of-range error |
| MediaPipe 33 landmarks + OpenCV frame processing (poster) | PARTIAL | Model loads and runs in VIDEO mode; landmark indices and skeleton connections are checked against MediaPipe's own enums. Detection on a **real person** is NOT TESTED (no licensed footage available). |
| Visibility checks, missing joints skipped (poster) | VERIFIED (synthetic) | Missing landmarks become "not measured" and are never scored as bad form |
| Smoothing to cut jitter (poster) | VERIFIED (synthetic) | Gap fill + rolling median + rolling mean, using time-based windows |
| Knee, hip, torso lean, shin lean, depth (poster, status report) | VERIFIED (synthetic) | Matches ground truth within 0.01 deg on clean synthetic data |
| Side-view orientation check (prompt) | VERIFIED (synthetic) | `check_orientation`; front view disables lean/depth scoring |
| Rep counter, 140 deg down / 160 deg up, measured at the lowest point (poster) | VERIFIED (synthetic) | Hysteresis + debounce + duration limits; 16/16 synthetic scenarios exact at 24-120 fps |
| Personalization from femur/torso and lower-leg/femur (poster) | VERIFIED (logic) | Bounded geometric heuristic; **not experimentally validated** (see [PERSONALIZATION.md](PERSONALIZATION.md)) |
| Entered ratios vs. ratios estimated from video (poster, status report) | VERIFIED (synthetic) | 25% tolerance warning |
| Start at 10, deduct per issue, explain each deduction (poster, proposal) | VERIFIED | `scoring.py`; deterministic; per-rep deductions with messages |
| Baseline vs. personalized comparison (prompt) | VERIFIED | Both computed for every rep; chart + table |
| Research-based guideline database (proposal) | PLANNED | Reference values live in `config.py`, tagged PROTOTYPE where they are not from team research. Billy's guideline research should replace them. |
| Elbow/arm evaluation (proposal mentions elbow angles) | PLANNED | Arm length is collected and checked against the video, but squat scoring does not use it |
| Calibration step before recording (proposal) | PARTIAL | Written recording instructions on stage 1; no interactive calibration |
| Skeleton overlay video (prompt, status report) | VERIFIED (plumbing) | WebM (VP8) encode verified; drawing verified on synthetic frames; a real-person overlay is NOT TESTED |
| Testing across camera angles, distances, lighting, clothing, participants (proposal, poster) | NOT TESTED | The poster says "We recorded lifts..." but no recordings or results were provided to this audit. `scripts/evaluate_videos.py` and [data/real_videos/README.md](../data/real_videos/README.md) implement the protocol. |
| ML model for ranking form, >75% accuracy (Status Report 2 goal) | PLANNED | Not built. There is no labeled dataset and no evaluation method yet. Do not report any accuracy figure. |
| AI-generated feedback (Status Report 2 research item) | PLANNED | Rules-based feedback only; comparison in [ROADMAP.md](ROADMAP.md) |

## 3. Discrepancies to resolve as a team

1. **Poster "Testing" bullet.** It says lifts were recorded under different conditions.
   If those videos exist, run `scripts/evaluate_videos.py` on them before the
   presentation. If they don't, reword the bullet as planned testing.
2. **"Research-based guideline ranges."** The current numbers are the poster's own
   100 deg depth tip plus PROTOTYPE values. Label them that way until Billy's research
   replaces them.
3. **">75% accuracy" goal.** This needs a definition first: what counts as a correct
   "form ranking", who labels the ground truth, and on how many videos.
4. **Depth definition.** The poster uses knee angle (< 100 deg). Powerlifting
   "parallel" compares the hip crease with the knee. The app reports both and only
   scores the knee-angle version.

## 4. Dependency and integration risks (found during setup)

| Risk | What happened | Mitigation |
|---|---|---|
| MediaPipe API change | MediaPipe 0.10.30+ (and every version installable on Python 3.13) removed `mp.solutions.pose`. Older tutorials, possibly including the team's webcam script, will break. | Uses the Tasks `PoseLandmarker` API plus a downloaded `.task` model |
| Windows Application Control | OpenCV 5.1 and pyarrow 26 DLLs were blocked by policy on the dev laptop | Pinned OpenCV 4.12.0.88 and pyarrow 21.0.0, which load. The security setting was not changed. |
| H.264 encoding | OpenCV's `avc1` writer hangs without OpenH264; browsers can't play `mp4v` | Overlay uses VP8 WebM (tested) |
| Streamlit network exposure | By default Streamlit listened on all interfaces and printed an external URL | `server.address = "localhost"` in `.streamlit/config.toml` |
| Streamlit telemetry | On by default | `browser.gatherUsageStats = false` |

## Merging with the team repository

When the team repo is available:

1. Commit this prototype on its own branch, `prototype/integration`, so nothing is
   overwritten.
2. **Kyle's pose code vs. `pose_detection.py`:** if his functions return 33 landmarks
   per frame, adapt them to fill a `LandmarkSequence` (see
   [DATA_SCHEMA.md](DATA_SCHEMA.md)). Everything after that point works unchanged.
   Run `pytest -m integration` to check.
3. **Kyle's analysis file vs. `squat_metrics.py` / `rep_counter.py`:** compare outputs
   on `data/synthetic/*.csv` and keep whichever version passes `tests/` (add his test
   cases too).
4. **Eric's UI vs. `app.py`:** keep his layout and visual decisions; the integration
   point is a single call to `analyze_video(path, measurements)`.
5. **Billy's guidelines:** replace PROTOTYPE values in `ratemyform/config.py` and record
   the sources next to each value.
6. Keep authorship in the git history, and list contributors in file docstrings
   where the team prefers.
