"""Repeatable evaluation on SYNTHETIC scenarios -> results/ (CSV, JSON, charts).

    python scripts/run_evaluation.py

Everything this script measures comes from running the real analysis code on
generated stick-figure landmark sequences with known ground truth. The
outputs are labeled SYNTHETIC. They show the logic behaves as designed under
controlled conditions (noise, dropouts, occlusion, frame rates); they are NOT
real-world accuracy results. For real recordings use
scripts/evaluate_videos.py.
"""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matplotlib.figure import Figure  # noqa: E402

from ratemyform import charts  # noqa: E402
from ratemyform.config import DEFAULT_CONFIG  # noqa: E402
from ratemyform.overlay import render_synthetic_frame  # noqa: E402
from ratemyform.personalization import BodyMeasurements, personalized_thresholds  # noqa: E402
from ratemyform.pipeline import analyze_sequence  # noqa: E402
from ratemyform.synthetic import default_measurements_for, generate_squat, scenarios  # noqa: E402

OUT = ROOT / "results"
CHARTS = OUT / "charts"
TIMING_REPEATS = 5


def environment() -> dict:
    pkgs = {}
    for name in ("mediapipe", "opencv-contrib-python", "numpy", "streamlit", "matplotlib"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": pkgs,
    }


def evaluate_scenario(name, spec, description) -> tuple[dict, object, object]:
    syn = generate_squat(spec)
    meas = default_measurements_for(spec)
    timings = []
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter()
        res = analyze_sequence(syn.sequence, meas, DEFAULT_CONFIG)
        timings.append((time.perf_counter() - t0) * 1000)

    truth = syn.truth
    measured_knee = [m.knee_angle_deg for m in res.rep_measurements]
    knee_err = None
    if res.rep_count == truth["expected_rep_count"] and res.rep_count:
        errs = [abs(m - t) for m, t in zip(measured_knee, truth["bottom_knee_angles_deg"]) if m is not None]
        knee_err = round(float(np.mean(errs)), 2) if errs else None
    fm = res.frame_metrics
    ok = np.isfinite(fm.smoothed["knee_angle"])
    frame_mae = round(float(np.mean(np.abs(fm.smoothed["knee_angle"][ok] - syn.true_knee_angle[ok]))), 2) if ok.any() else None
    lean_ok = np.isfinite(fm.smoothed["torso_lean"])
    lean_mae = round(float(np.mean(np.abs(fm.smoothed["torso_lean"][lean_ok] - syn.true_torso_lean[lean_ok]))), 2) if lean_ok.any() else None

    row = {
        "data": "SYNTHETIC",
        "scenario": name,
        "description": description,
        "fps": spec.fps,
        "frames": syn.sequence.num_frames,
        "duration_s": round(syn.sequence.duration_s, 2),
        "expected_reps": truth["expected_rep_count"],
        "detected_reps": res.rep_count,
        "rep_count_correct": res.rep_count == truth["expected_rep_count"],
        "uncounted": ";".join(u.kind for u in res.rep_detection.uncounted),
        "bottom_knee_abs_error_deg": knee_err,
        "frame_knee_mae_deg": frame_mae,
        "frame_torso_lean_mae_deg": lean_mae,
        "orientation": res.orientation.label,
        "low_confidence_reps": sum(m.low_confidence for m in res.rep_measurements),
        "baseline_score": res.overall_baseline,
        "personalized_score": res.overall_personalized,
        "torso_lean_adjustment_deg": res.personalized.torso_lean_adjustment_deg,
        "warnings": len(res.warnings),
        "analysis_ms_median": round(statistics.median(timings), 1),
    }
    return row, res, syn


def chart_rep_counts(rows) -> Figure:
    fig = Figure(figsize=(9, 3.8))
    ax = fig.subplots()
    correct = sum(r["rep_count_correct"] for r in rows)
    charts._style(ax, f"SYNTHETIC scenarios: expected vs. detected reps ({correct} of {len(rows)} exact)", "", "Reps")
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, [r["expected_reps"] for r in rows], width=w, color=charts.INK_2, label="Expected (ground truth)")
    ax.bar(x + w / 2 + 0.01, [r["detected_reps"] for r in rows], width=w, color=charts.BLUE, label="Detected by RateMyForm")
    ax.set_xticks(x, [r["scenario"] for r in rows], rotation=40, ha="right", fontsize=7.5)
    ax.set_ylim(0, 6.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=8, frameon=False)
    return fig


def chart_noise_timeline(res, syn) -> Figure:
    fm = res.frame_metrics
    fig = Figure(figsize=(9, 3.6))
    ax = fig.subplots()
    charts._style(ax, "SYNTHETIC noisy input: per-frame vs. smoothed knee angle", "Time (s)", "Knee angle (deg)")
    for rep in res.rep_detection.reps:
        ax.axvspan(rep.start_time_s, rep.end_time_s, color=charts.REP_BAND, alpha=0.6, linewidth=0)
    ax.plot(fm.times_s, fm.raw["knee_angle"], ".", color=charts.MUTED, markersize=3, label="Per-frame (noise + spikes)")
    ax.plot(fm.times_s, syn.true_knee_angle, "-", color=charts.INK, linewidth=1, label="Ground truth")
    ax.plot(fm.times_s, fm.smoothed["knee_angle"], "-", color=charts.BLUE, linewidth=2, label="Smoothed")
    for y in (DEFAULT_CONFIG.reps.up_threshold_deg, DEFAULT_CONFIG.reps.down_threshold_deg):
        ax.axhline(y, color=charts.MUTED, linestyle="--", linewidth=1)
        ax.text(1.01, y, f"{y:.0f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=charts.INK_2)
    ax.set_ylim(50, 195)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    return fig


def chart_personalization_curve() -> Figure:
    """Torso-lean limit as a function of femur/torso ratio (model + bound)."""
    ratios = np.linspace(0.6, 1.2, 61)
    base = DEFAULT_CONFIG.scoring.torso_lean_limit_deg
    limits = [
        personalized_thresholds(BodyMeasurements(50.0, 50.0 * r, 50.0 * r, 57.0), DEFAULT_CONFIG.scoring, DEFAULT_CONFIG.personalization).torso_lean_limit_deg
        for r in ratios
    ]
    fig = Figure(figsize=(7, 3.6))
    ax = fig.subplots()
    charts._style(ax, "How personalization moves the torso-lean limit", "Femur / torso ratio (lower-leg/femur = 1.0)", "Torso-lean limit (deg)")
    ax.axhline(base, color=charts.BLUE, linestyle="--", linewidth=1.5, label=f"Baseline limit ({base:.0f}, everyone)")
    ax.plot(ratios, limits, "-", color=charts.ORANGE, linewidth=2, label="Personalized limit (bounded to +/-10)")
    ax.axvline(DEFAULT_CONFIG.personalization.reference_femur_to_torso, color=charts.MUTED, linewidth=1, linestyle=":")
    ax.text(DEFAULT_CONFIG.personalization.reference_femur_to_torso, 33.5, " reference\n proportions", fontsize=7.5, color=charts.INK_2)
    ax.set_ylim(32, 58)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    return fig


def chart_baseline_vs_personalized(rows) -> Figure:
    pick = [r for r in rows if r["scenario"] in ("clean_5_reps", "forward_lean_long_femur", "forward_lean_short_femur", "shallow_reps", "chest_drop")]
    fig = Figure(figsize=(8, 3.6))
    ax = fig.subplots()
    charts._style(ax, "SYNTHETIC: overall score, baseline vs. personalized", "", "Score (0-10)")
    x = np.arange(len(pick))
    w = 0.36
    for off, key, color, label in ((-w / 2 - 0.01, "baseline_score", charts.BLUE, "Baseline (same for everyone)"), (w / 2 + 0.01, "personalized_score", charts.ORANGE, "Personalized")):
        vals = [r[key] or 0 for r in pick]
        bars = ax.bar(x + off, vals, width=w, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.5, color=charts.INK_2)
    ax.set_xticks(x, [r["scenario"] for r in pick], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 11.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8, frameon=False)
    return fig


def export_sample(name: str, res, syn) -> None:
    """Write a readable per-frame CSV of the analyzed-side landmarks + ground truth."""
    from ratemyform.landmarks import SIDE_LANDMARKS

    seq = syn.sequence
    side = res.frame_metrics.side
    idx = SIDE_LANDMARKS[side]
    parts = ("shoulder", "hip", "knee", "ankle")
    path = ROOT / "data" / "synthetic" / f"{name}_landmarks.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["data", "time_s", "detected"]
        for p in parts:
            header += [f"{p}_x_px", f"{p}_y_px", f"{p}_visibility", f"{p}_world_x_m", f"{p}_world_y_m", f"{p}_world_z_m"]
        header += ["true_knee_angle_deg", "true_torso_lean_deg", "measured_knee_angle_deg", "measured_torso_lean_deg"]
        w.writerow(header)
        for i in range(seq.num_frames):
            row = ["SYNTHETIC", round(float(seq.timestamps_s[i]), 4), bool(seq.detected[i])]
            for p in parts:
                j = idx[p]
                row += [round(float(v), 2) for v in seq.image_xy[i, j]] + [round(float(seq.visibility[i, j]), 3)]
                row += [round(float(v), 4) for v in seq.world_xyz[i, j]]
            sm = res.frame_metrics.smoothed
            row += [
                round(float(syn.true_knee_angle[i]), 2),
                round(float(syn.true_torso_lean[i]), 2),
                None if not np.isfinite(sm["knee_angle"][i]) else round(float(sm["knee_angle"][i]), 2),
                None if not np.isfinite(sm["torso_lean"][i]) else round(float(sm["torso_lean"][i]), 2),
            ]
            w.writerow(row)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)
    rows, keep = [], {}
    for name, (spec, desc) in scenarios().items():
        row, res, syn = evaluate_scenario(name, spec, desc)
        rows.append(row)
        keep[name] = (res, syn)
        print(f"{name:26s} expected={row['expected_reps']} detected={row['detected_reps']} "
              f"base={row['baseline_score']} pers={row['personalized_score']} {row['analysis_ms_median']} ms")

    with open(OUT / "synthetic_evaluation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "label": "SYNTHETIC - generated stick-figure data with known ground truth; not real people",
        "environment": environment(),
        "scenarios": len(rows),
        "rep_count_exact": sum(r["rep_count_correct"] for r in rows),
        "expected_reps_total": sum(r["expected_reps"] for r in rows),
        "detected_reps_total": sum(r["detected_reps"] for r in rows),
        "median_analysis_ms": round(statistics.median(r["analysis_ms_median"] for r in rows), 1),
        "rows": rows,
    }
    (OUT / "synthetic_evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Example full reports (what the app exports) for two scenarios.
    for name in ("forward_lean_long_femur", "occluded_shoulder"):
        (OUT / f"example_report_{name}.json").write_text(json.dumps(keep[name][0].to_dict(), indent=2), encoding="utf-8")

    figures = {
        "rep_detection_synthetic.png": chart_rep_counts(rows),
        "noise_smoothing_synthetic.png": chart_noise_timeline(*keep["noisy_jitter"]),
        "personalization_curve.png": chart_personalization_curve(),
        "baseline_vs_personalized_synthetic.png": chart_baseline_vs_personalized(rows),
        "knee_timeline_long_femur_synthetic.png": charts.knee_angle_timeline(keep["forward_lean_long_femur"][0], " (SYNTHETIC)"),
        "rep_scores_long_femur_synthetic.png": charts.score_comparison(keep["forward_lean_long_femur"][0]),
        "torso_lean_long_femur_synthetic.png": charts.bottom_metrics_chart(keep["forward_lean_long_femur"][0]),
    }
    for filename, fig in figures.items():
        (CHARTS / filename).write_bytes(charts.to_png(fig))

    for name in ("clean_5_reps", "noisy_jitter", "occluded_shoulder"):
        export_sample(name, *keep[name])

    # Synthetic skeleton key frame (clearly labeled) for slides.
    res, syn = keep["forward_lean_long_femur"]
    rep = res.rep_detection.reps[0]
    (CHARTS / "skeleton_keyframe_synthetic.png").write_bytes(
        render_synthetic_frame(syn.sequence, res.frame_metrics, rep.bottom_index, 0, DEFAULT_CONFIG)
    )

    print(f"\n{summary['rep_count_exact']}/{summary['scenarios']} scenarios counted exactly "
          f"({summary['detected_reps_total']}/{summary['expected_reps_total']} reps). Outputs in {OUT}")
    return 0 if summary["rep_count_exact"] == summary["scenarios"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
