"""Evaluate RateMyForm on REAL recordings listed in a manifest CSV.

    python scripts/evaluate_videos.py data/real_videos/manifest.csv

Manifest columns (see data/real_videos/manifest_template.csv):
    video            path relative to the manifest file
    true_reps        reps counted manually by a person watching the video
    camera_angle     e.g. side / 45deg / front
    distance_m       approximate camera distance
    lighting         e.g. bright / dim / backlit
    clothing         e.g. fitted / loose
    torso_cm, femur_cm, lower_leg_cm, arm_cm   participant measurements
    notes            free text

Outputs (git-ignored, may describe real people):
    results/real/video_evaluation.csv   one row per video
    results/real/video_evaluation.json  summary + environment
Only record and use videos of people who agreed to be recorded and analyzed.
"""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratemyform.config import DEFAULT_CONFIG  # noqa: E402
from ratemyform.personalization import BodyMeasurements  # noqa: E402
from ratemyform.pipeline import analyze_video  # noqa: E402

REQUIRED = ["video", "true_reps", "torso_cm", "femur_cm", "lower_leg_cm", "arm_cm"]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    manifest = Path(sys.argv[1])
    rows_in = list(csv.DictReader(manifest.open(encoding="utf-8")))
    missing = [c for c in REQUIRED if rows_in and c not in rows_in[0]]
    if not rows_in or missing:
        print(f"ERROR: manifest is empty or missing columns {missing}", file=sys.stderr)
        return 2

    rows_out = []
    for r in rows_in:
        path = (manifest.parent / r["video"]).resolve()
        meas = BodyMeasurements(float(r["torso_cm"]), float(r["femur_cm"]), float(r["lower_leg_cm"]), float(r["arm_cm"]))
        t0 = time.perf_counter()
        try:
            res = analyze_video(path, meas, DEFAULT_CONFIG, make_overlay=False)
            error = ""
        except Exception as exc:  # noqa: BLE001 - record and continue with the next video
            res, error = None, f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - t0
        true_reps = int(r["true_reps"])
        detected = res.rep_count if res else None
        frames = res.processing["frames_analyzed"] if res else None
        rows_out.append(
            {
                "video": r["video"],
                "camera_angle": r.get("camera_angle", ""),
                "distance_m": r.get("distance_m", ""),
                "lighting": r.get("lighting", ""),
                "clothing": r.get("clothing", ""),
                "true_reps": true_reps,
                "detected_reps": detected,
                "abs_rep_error": None if detected is None else abs(detected - true_reps),
                "exact": detected == true_reps,
                "detection_rate": res.processing["detection_rate"] if res else None,
                "orientation": res.orientation.label if res else None,
                "low_confidence_reps": sum(m.low_confidence for m in res.rep_measurements) if res else None,
                "personalized_score": res.overall_personalized if res else None,
                "baseline_score": res.overall_baseline if res else None,
                "frames": frames,
                "processing_s": round(seconds, 2),
                "ms_per_frame": round(1000 * seconds / frames, 1) if frames else None,
                "warnings": len(res.warnings) if res else None,
                "error": error,
            }
        )
        print(f"{r['video']}: true={true_reps} detected={detected} ({seconds:.1f} s) {error}")

    out = ROOT / "results" / "real"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "video_evaluation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    ok = [x for x in rows_out if x["detected_reps"] is not None]
    summary = {
        "label": "REAL recordings (manually counted ground truth)",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor()},
        "videos": len(rows_out),
        "failed_to_process": len(rows_out) - len(ok),
        "exact_rep_count_videos": sum(x["exact"] for x in ok),
        "mean_abs_rep_error": round(statistics.mean(x["abs_rep_error"] for x in ok), 2) if ok else None,
        "mean_detection_rate": round(statistics.mean(x["detection_rate"] for x in ok), 3) if ok else None,
        "median_ms_per_frame": statistics.median(x["ms_per_frame"] for x in ok) if ok else None,
        "by_condition": {},
        "rows": rows_out,
    }
    for key in ("camera_angle", "lighting", "clothing"):
        groups: dict[str, list] = {}
        for x in ok:
            groups.setdefault(x[key] or "unspecified", []).append(x)
        summary["by_condition"][key] = {
            g: {"videos": len(v), "exact": sum(i["exact"] for i in v), "mean_abs_rep_error": round(statistics.mean(i["abs_rep_error"] for i in v), 2)}
            for g, v in groups.items()
        }
    (out / "video_evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
