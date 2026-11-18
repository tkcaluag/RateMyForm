"""Analyze one real squat video from the command line (no browser needed).

    python scripts/analyze_video.py my_squat.mp4 --torso 50 --femur 43 --lower-leg 42 --arm 58 --unit cm

Writes to results/real/<video name>/ (git-ignored, because it can contain
images of a real person):
    report.json, reps.csv, timeseries.csv, overlay.webm, rep_<n>_bottom.png,
    knee_timeline.png, rep_scores.png, torso_lean.png
Everything runs locally; nothing is uploaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratemyform import charts  # noqa: E402
from ratemyform.config import DEFAULT_CONFIG  # noqa: E402
from ratemyform.pipeline import analyze_video  # noqa: E402
from ratemyform.pose_detection import probe_video  # noqa: E402
from ratemyform.validation import validate_measurements, validate_upload, validate_video_properties  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--torso", type=float, required=True)
    p.add_argument("--femur", type=float, required=True)
    p.add_argument("--lower-leg", type=float, required=True)
    p.add_argument("--arm", type=float, required=True)
    p.add_argument("--unit", choices=["cm", "in"], default="cm")
    p.add_argument("--out", type=Path, default=None, help="output folder (default results/real/<video stem>)")
    p.add_argument("--no-overlay", action="store_true")
    args = p.parse_args()

    if not args.video.is_file():
        print(f"ERROR: {args.video} not found", file=sys.stderr)
        return 2
    upload = validate_upload(args.video.name, args.video.stat().st_size, DEFAULT_CONFIG.limits)
    meas = validate_measurements(
        {"torso": args.torso, "femur": args.femur, "lower_leg": args.lower_leg, "arm": args.arm}, args.unit, DEFAULT_CONFIG.limits
    )
    errors = upload.errors + meas.errors
    if not errors:
        info = probe_video(args.video)
        props = validate_video_properties(info.fps, info.frame_count, info.width, info.height, DEFAULT_CONFIG.limits)
        errors += props.errors
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for w in meas.warnings:
        print(f"WARNING: {w}")

    def progress(f: float, msg: str) -> None:
        print(f"\r{f:5.0%} {msg[:60]:60s}", end="", flush=True)

    result = analyze_video(args.video, meas.measurements, DEFAULT_CONFIG, progress=progress, make_overlay=not args.no_overlay)
    print()

    out = args.out or ROOT / "results" / "real" / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    write_csv(out / "reps.csv", result.reps_table())
    write_csv(out / "timeseries.csv", result.timeseries_rows())
    if result.overlay_video:
        (out / "overlay.webm").write_bytes(result.overlay_video)
    for n, png in result.key_frames.items():
        (out / f"rep_{n}_bottom.png").write_bytes(png)
    (out / "knee_timeline.png").write_bytes(charts.to_png(charts.knee_angle_timeline(result)))
    if result.comparison:
        (out / "rep_scores.png").write_bytes(charts.to_png(charts.score_comparison(result)))
        (out / "torso_lean.png").write_bytes(charts.to_png(charts.bottom_metrics_chart(result)))

    print(f"Reps counted: {result.rep_count}")
    print(f"Personalized score: {result.overall_personalized}  Baseline score: {result.overall_baseline}")
    print(f"Person detected in {result.processing['detection_rate']:.0%} of frames; camera: {result.orientation.label}")
    print(f"Processing: {result.processing.get('total_seconds')} s for {result.processing['frames_analyzed']} frames")
    for tip in result.feedback:
        print(f"- {tip}")
    for w in result.warnings:
        print(f"! {w}")
    print(f"Saved outputs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
