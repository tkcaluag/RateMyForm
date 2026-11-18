"""Matplotlib charts for the results page and the evaluation report.

Colors follow a validated two-series palette (blue = baseline, orange =
personalized; checked for color-vision-deficiency separation). Every series
is also direct-labeled or in a legend, so color is never the only cue.
"""
from __future__ import annotations

import io

import numpy as np
from matplotlib.figure import Figure  # object API: safe when Streamlit runs sessions in threads

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # series 1: baseline / measured signal
ORANGE = "#eb6834"  # series 2: personalized
REP_BAND = "#cde2fb"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return buf.getvalue()


def knee_angle_timeline(result, title_suffix: str = ""):
    """Knee angle over time with rep thresholds and counted reps shaded."""
    fm = result.frame_metrics
    t = fm.times_s
    raw = fm.raw["knee_angle"]
    smooth = fm.smoothed["knee_angle"]
    fig = Figure(figsize=(9, 3.6))
    ax = fig.subplots()
    _style(ax, f"Knee angle over time{title_suffix}", "Time (s)", "Knee angle (deg)")
    for rep in result.rep_detection.reps:
        ax.axvspan(rep.start_time_s, rep.end_time_s, color=REP_BAND, alpha=0.6, linewidth=0)
        y = np.nanmin(smooth[rep.start_index : rep.end_index + 1])
        ax.annotate(
            f"Rep {rep.rep_number}",
            (rep.bottom_time_s, y),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_2,
        )
    ax.plot(t, raw, ".", color=MUTED, markersize=2.5, label="Per-frame estimate")
    ax.plot(t, smooth, "-", color=BLUE, linewidth=2, label="Smoothed")
    cfg_up = result.config.reps.up_threshold_deg
    cfg_down = result.config.reps.down_threshold_deg
    target = result.baseline.depth_target_knee_angle_deg
    for y, label in ((cfg_up, f"{cfg_up:.0f}: back to standing"), (cfg_down, f"{cfg_down:.0f}: rep starts"), (target, f"{target:.0f}: depth reference")):
        ax.axhline(y, color=MUTED, linestyle="--", linewidth=1)
        # Label in the right margin so it never sits on top of the curve.
        ax.text(1.01, y, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.5, color=INK_2)
    finite = smooth[np.isfinite(smooth)]
    low = min(60.0, float(finite.min()) - 15) if finite.size else 60.0
    ax.set_ylim(low, 190)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    return fig


def score_comparison(result):
    """Grouped bars: baseline vs personalized score for each rep."""
    reps = [c["rep_number"] for c in result.comparison]
    base = [c["baseline_score"] for c in result.comparison]
    pers = [c["personalized_score"] for c in result.comparison]
    fig = Figure(figsize=(7, 3.4))
    ax = fig.subplots()
    _style(ax, "Score per rep: baseline vs. personalized", "Rep", "Score (0-10)")
    x = np.arange(len(reps))
    w = 0.36
    for offset, values, color, label in ((-w / 2 - 0.01, base, BLUE, "Baseline (same for everyone)"), (w / 2 + 0.01, pers, ORANGE, "Personalized (your proportions)")):
        heights = [0 if v is None else v for v in values]
        bars = ax.bar(x + offset, heights, width=w, color=color, label=label)
        for bar, v in zip(bars, values):
            text = "n/a" if v is None else f"{v:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, text, ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, [str(r) for r in reps])
    ax.set_ylim(0, 11.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    return fig


def bottom_metrics_chart(result):
    """Torso lean at the bottom of each rep vs. both reference limits."""
    reps = [m.rep_number for m in result.rep_measurements]
    lean = [np.nan if m.torso_lean_deg is None or not m.reliable.get("torso_lean") else m.torso_lean_deg for m in result.rep_measurements]
    fig = Figure(figsize=(7, 3.4))
    ax = fig.subplots()
    _style(ax, "Torso lean at the bottom of each rep", "Rep", "Torso lean from vertical (deg)")
    x = np.arange(len(reps))
    # Neutral bars: blue/orange are reserved for the baseline/personalized limits.
    ax.bar(x, np.nan_to_num(lean), width=0.5, color=INK_2, label="Measured lean (reliable reps only)")
    for xi, v in zip(x, lean):
        ax.text(xi, (0 if np.isnan(v) else v) + 0.8, "not scored" if np.isnan(v) else f"{v:.0f}", ha="center", fontsize=7.5, color=INK_2)
    b = result.baseline.torso_lean_limit_deg
    p = result.personalized.torso_lean_limit_deg
    ax.axhline(b, color=BLUE, linestyle="--", linewidth=1.5, label=f"Baseline limit ({b:.0f})")
    ax.axhline(p, color=ORANGE, linestyle="--", linewidth=1.5, label=f"Personalized limit ({p:.1f})")
    ax.set_xticks(x, [str(r) for r in reps])
    top = max([b, p] + [v for v in lean if not np.isnan(v)])
    ax.set_ylim(0, top + 10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    return fig
