# Real test videos (not committed)

Put real squat recordings here to evaluate RateMyForm on actual footage.
Video files in this folder are ignored by git because they show real people.

## Consent and privacy

- Only record people who agree to be recorded **and** to have the video analyzed.
- Avoid capturing bystanders (see Tracy's Project Considerations).
- Do not upload these videos to external services. RateMyForm processes them locally.
- Delete recordings when the evaluation is finished if participants ask.

## Suggested recording plan (from our proposal's testing procedure)

Record each participant doing 5 squats per condition, starting and ending standing:

| Variable | Levels |
|---|---|
| Camera angle | side (90 deg), angled (~45 deg), front |
| Distance | ~2 m, ~3 m, ~4 m |
| Lighting | bright, dim, backlit |
| Clothing | fitted, loose/oversized |

Count the reps manually while watching each video and enter them in `true_reps`.

## Running the evaluation

1. Copy `manifest_template.csv` to `manifest.csv` and add one row per video.
2. Run: `python scripts/evaluate_videos.py data/real_videos/manifest.csv`
3. Results are written to `results/real/video_evaluation.csv` and `.json`.

These results are the only ones that can support real-world accuracy claims.
