# RateMyForm

**Personalized squat form feedback from a smartphone video.**
CPSC 490 senior project, California State University, Fullerton.
Team: Tristan Caluag, Billy Leong, Eric Tran, Tracy Tran, Kyle Ly.

You upload a side-view video of your squats and enter four body measurements
(torso, femur, lower leg, arm). RateMyForm runs MediaPipe Pose with OpenCV on your
computer, counts reps with a hysteresis state machine, measures each rep at its
lowest point, and gives an explainable 0-10 score. The score is computed twice: once
against the same reference ranges for everyone ("baseline"), and once with a
torso-lean limit adjusted for your body proportions ("personalized").

> **Prototype status.** The scores are prototype estimates. They are not a
> validated injury-risk, medical, or clinical assessment. Only squats are
> supported. Accuracy on real videos has **not been measured yet** (see
> [docs/TESTING.md](docs/TESTING.md)).

## Quick start (Windows, tested)

Requires Python 3.13 (tested with 3.13.17). No GPU, API keys, or paid services.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

```bash
.venv\Scripts\python.exe scripts\download_model.py
```

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at http://localhost:8501. On macOS/Linux, use `.venv/bin/python`
instead of `.venv\Scripts\python.exe` (not tested on those systems).

`download_model.py` fetches the official MediaPipe Pose Landmarker model
(`pose_landmarker_full.task`, about 9.4 MB, Apache 2.0) from Google's model
storage and records its SHA-256 in `models/MODEL_PROVENANCE.md`.

No video yet? On the upload page, open **"Try a SYNTHETIC demo"**. It runs the
real analysis on a computer-generated stick figure, and the results are clearly
labeled as synthetic.

## Other commands

| Task | Command |
|---|---|
| Run all automated tests | `.venv\Scripts\python.exe -m pytest` |
| Only fast logic tests | `.venv\Scripts\python.exe -m pytest -m "not integration and not ui"` |
| Synthetic evaluation (CSV/JSON/charts in `results/`) | `.venv\Scripts\python.exe scripts\run_evaluation.py` |
| Analyze one real video from the terminal | `.venv\Scripts\python.exe scripts\analyze_video.py my_squat.mp4 --torso 50 --femur 43 --lower-leg 42 --arm 58` |
| Evaluate real recordings vs. manual rep counts | `.venv\Scripts\python.exe scripts\evaluate_videos.py data\real_videos\manifest.csv` |

## Project layout

```
app.py                      Streamlit UI: 3 stages (exercise -> upload+measurements -> results)
ratemyform/
  config.py                 every threshold, tagged TEAM_METHOD / MODEL_CARD / PROTOTYPE
  landmarks.py              33 MediaPipe landmark indices + LandmarkSequence arrays
  pose_detection.py         OpenCV video reading + MediaPipe PoseLandmarker (VIDEO mode)
  geometry.py               joint angle and lean math
  smoothing.py              gap filling, rolling median/mean (time-based windows)
  squat_metrics.py          per-frame knee/hip angle, torso/shin lean, depth, side-view check
  rep_counter.py            hysteresis state machine (140 / 160 deg)
  rep_analysis.py           bottom-of-rep measurements + reliability per metric
  personalization.py        body ratios, bounded torso-lean adjustment, ratio checks
  scoring.py                explainable 10-point deductions (baseline and personalized)
  feedback.py               plain-language tips
  pipeline.py               analyze_video() / analyze_sequence() -> AnalysisResult (+JSON)
  overlay.py                skeleton drawing, WebM overlay video, key frames
  charts.py                 matplotlib charts used by the app and reports
  synthetic.py              SYNTHETIC squat generator with ground truth (tests/demo only)
  validation.py             measurement, unit, upload, and video checks
scripts/                    download_model, run_evaluation, analyze_video, evaluate_videos
tests/                      166 automated tests (pytest)
data/synthetic/             sample SYNTHETIC landmark CSVs
data/real_videos/           where real recordings go (git-ignored) + manifest template
results/                    generated evaluation outputs and charts
docs/                       audit, architecture, schema, personalization, testing, limitations,
                            roadmap, learning guide, presentation material, original submissions
```

## Documentation

- [docs/AUDIT.md](docs/AUDIT.md): our CPSC 490 documents vs. what is actually built
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the pieces fit, with diagrams
- [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md): inputs, the JSON report, and CSV exports
- [docs/PERSONALIZATION.md](docs/PERSONALIZATION.md): the body-ratio model and its limits
- [docs/TESTING.md](docs/TESTING.md): test commands, results, and what is not tested
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md): assumptions and known limitations
- [docs/ROADMAP.md](docs/ROADMAP.md): future development
- [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md): study guide with quiz questions
- [docs/presentation/RESULTS_AND_OUTCOMES.md](docs/presentation/RESULTS_AND_OUTCOMES.md): midterm Results & Outcomes material

## Privacy

- Videos are processed **on this computer**. The temporary copy is deleted right after
  analysis, and nothing is sent to an external service.
- The app only listens on `localhost`, and Streamlit usage statistics are turned off
  (`.streamlit/config.toml`).
- Exported reports contain measurements and scores, not the video.
- Real recordings in `data/real_videos/` and outputs in `results/real/` are git-ignored.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: DLL load failed ... An Application Control policy has blocked this file` | Windows Smart App Control blocked a new native wheel. The versions pinned in `requirements.txt` (OpenCV 4.12.0.88, pyarrow 21.0.0) load on the dev machine. Do not disable the security feature; use the pinned versions. |
| `MediaPipe model not found` | Run `scripts\download_model.py`. |
| `module 'mediapipe' has no attribute 'solutions'` in old code | MediaPipe 0.10.30+ removed the legacy API; this project uses the Tasks API. |
| Overlay video does not play | The overlay is VP8 WebM. Chrome, Edge, and Firefox play it; Safari support was not tested. |
| No reps counted | The knee must go below 140 deg and back above 160 deg, so start and end each rep standing tall. The camera must see the whole body from the side. |

## Team attribution

The work division follows Status Report 2: Tristan (MediaPipe computer vision), Kyle
(pose detection module, squat measurements, rep counting, proportion comparison),
Eric (UI and video upload), Billy (exercise guideline research and data), and Tracy
(software research, requirements, frontend planning, user workflow, integration).
The team's own source code was **not available** when this prototype was written, so
`ratemyform/` is a fresh implementation of the interfaces described in our reports.
See [docs/AUDIT.md](docs/AUDIT.md#merging-with-the-team-repository) for how to merge
it with teammates' code without losing their work.

## References

### Project background (from our CPSC 490 proposal)

[1] A. Patil, D. Rao, K. Utturwar, T. Shelke, and E. Sarda, "Body posture detection and motion tracking using AI for medical exercises and recommendation system," *ITM Web of Conferences*, vol. 44, Art. no. 03043, 2022, doi: 10.1051/itmconf/20224403043.

[2] K. M., M. Kandala, V. G. S., V. N., and S. Palaniswamy, "AI-based posture correction, real-time exercise tracking and feedback using pose estimation technique," in *Proc. 2024 Int. Conf. Communication, Control, and Intelligent Systems (CCIS)*, Mathura, India, 2024, pp. 1-6, doi: 10.1109/CCIS63231.2024.10932054.

[3] Prateek, A. Tanvir, and R. Brindha, "AI-powered workout analysis application for posture feedback and repitition grading," in *Proc. 2024 Second Int. Conf. Inventive Computing and Informatics (ICICI)*, Bangalore, India, 2024, pp. 108-111, doi: 10.1109/ICICI62254.2024.00027.

[4] H. Jubair, R. T. Chowdhury, F. Akter, I. S. Sami, N. Nawal, S. T. Arshi, and P. Shil, "Machine learning for real-time exercise correction and injury prevention: A systematic review," *Authorea Preprints*, 2025, doi: 10.22541/au.174466167.73334339/v1.

[5] J. Medhat and Z. Taha, "AI virtual gym trainer: Using deep learning for real-time exercise guidance and form correction," in *Proc. 2025 Intelligent Methods, Systems, and Applications (IMSA)*, Giza, Egypt, 2025, pp. 623-628, doi: 10.1109/IMSA65733.2025.11167265.

[6] J. Singh, A. Parnami, N. Dagar, and M. S. Adhikari, "AI-powered real-time workout pose analysis and correction system," in *Proc. 2026 5th Int. Conf. Innovative Practices in Technology and Management (ICIPTM)*, Noida, India, 2026, pp. 1-6, doi: 10.1109/ICIPTM69057.2026.11465414.

[7] M. J.-Y. Tung, G. A. Lantz, A. D. Lopes, and L. Berglund, "Injuries in weightlifting and powerlifting: An updated systematic review," *BMJ Open Sport & Exercise Medicine*, vol. 10, no. 4, Art. no. e001884, 2024, doi: 10.1136/bmjsem-2023-001884.

[8] K. Golshani, M. E. Cinque, P. O'Halloran, K. Softness, L. Keeling, and J. R. Macdonell, "Upper extremity weightlifting injuries: Diagnosis and management," *Journal of Orthopaedics*, vol. 15, no. 1, pp. 24-27, 2018, doi: 10.1016/j.jor.2017.11.005.

[9] M. Ramzan, "Full body detection with OpenCV and MediaPipe," *Medium*. [Online]. Available: https://medium.com/@iamramzan/full-body-detection-with-opencv-and-mediapipe-3b80cfc131ec. [Accessed: Sep. 14, 2026].

### Pose estimation model (used by `pose_detection.py`)

[10] Google, "MediaPipe Pose Landmarker," Google AI Edge documentation. [Online]. Available: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker. [Accessed: Nov. 17, 2026].

[11] V. Bazarevsky, I. Grishchenko, and E. G. Bazavan, "Model card: MediaPipe BlazePose GHUM 3D," Google, Apr. 2021. Licensed under Apache License 2.0. [Online]. Available: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf. [Accessed: Nov. 17, 2026].

[12] V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and M. Grundmann, "BlazePose: On-device real-time body pose tracking," presented at the CVPR Workshop on Computer Vision for Augmented and Virtual Reality, Seattle, WA, USA, 2020, arXiv:2006.10204.

### Squat biomechanics (used in [docs/PERSONALIZATION.md](docs/PERSONALIZATION.md))

[13] A. C. Fry, J. C. Smith, and B. K. Schilling, "Effect of knee position on hip and knee torques during the barbell squat," *J. Strength Cond. Res.*, vol. 17, no. 4, pp. 629-633, Nov. 2003, doi: 10.1519/1533-4287(2003)017<0629:eokpoh>2.0.co;2.

[14] S. Lorenzetti, M. Stoop, T. Ukelo, H. Gerber, A. Stacoff, and E. Stüssi, "Comparison of angles and the corresponding moments in knee and hip during restricted and unrestricted squats," in *Proc. 27th Int. Conf. Biomechanics in Sports (ISBS)*, 2009. [Online]. Available: https://ojs.ub.uni-konstanz.de/cpa/article/view/3248. [Accessed: Nov. 17, 2026]. (Five subjects; found the same direction as [13] but states that its tenfold hip-torque increase "was not observed.")

*To verify (not yet confirmed for this project):* body-segment lengths as fractions of height attributed to R. Drillis and R. Contini, "Body segment parameters," School of Engineering and Science, New York Univ., New York, NY, USA, Tech. Rep. 1166.03, 1966, as reproduced in D. A. Winter, *Biomechanics and Motor Control of Human Movement*. The prototype's reference body proportions in `ratemyform/config.py` should be checked against this source. The report details come from works that cite it; the report itself was not obtained.

### Software

| Library | Version | License | Website |
|---|---|---|---|
| MediaPipe | 1.2.3 | Apache 2.0 | https://github.com/google/mediapipe |
| OpenCV (opencv-contrib-python) | 4.12.0.88 | Apache 2.0 | https://github.com/opencv/opencv-python |
| NumPy | 2.2.6 | BSD | https://numpy.org |
| Streamlit | 1.67.0 | Apache 2.0 | https://streamlit.io |
| pandas | 3.1.1 | BSD 3-Clause | https://pandas.pydata.org |
| Apache Arrow (pyarrow) | 21.0.0 | Apache 2.0 | https://arrow.apache.org |
| Matplotlib | 3.11.2 | Matplotlib license (PSF-based) | https://matplotlib.org |
| pytest | 9.2.0 | MIT | https://docs.pytest.org |

License information was read from each installed package's metadata.
