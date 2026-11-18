"""Skeleton overlay drawing and annotated video export (OpenCV).

The overlay video is written as WebM (VP8) because browsers cannot play
OpenCV's default MP4 codec (mp4v), and this OpenCV build's H.264 encoder
hangs on Windows without the separate OpenH264 library (tested; see
docs/TESTING.md). Chrome, Edge, and Firefox play VP8 WebM.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import AnalysisConfig
from .landmarks import POSE_CONNECTIONS, SIDE_LANDMARKS, LandmarkSequence
from .pose_detection import iter_frames
from .rep_counter import RepDetectionResult
from .squat_metrics import FrameMetrics

# BGR colors
COLOR_BONE = (235, 235, 235)
COLOR_BONE_LOW = (130, 130, 130)
COLOR_JOINT = (60, 200, 255)  # analyzed-side joints
COLOR_JOINT_LOW = (110, 110, 110)
COLOR_TEXT = (255, 255, 255)


def draw_pose(
    frame: np.ndarray,
    xy: np.ndarray,
    visibility: np.ndarray,
    side: str | None,
    min_vis: float,
) -> np.ndarray:
    """Draw the 33-landmark skeleton. Low-visibility parts are drawn grey."""
    out = frame
    if not np.isfinite(xy).any():
        return out
    scale = max(1, int(round(max(out.shape[:2]) / 480)))
    for a, b in POSE_CONNECTIONS:
        if not (np.isfinite(xy[a]).all() and np.isfinite(xy[b]).all()):
            continue
        good = visibility[a] >= min_vis and visibility[b] >= min_vis
        p1 = tuple(int(round(v)) for v in xy[a])
        p2 = tuple(int(round(v)) for v in xy[b])
        cv2.line(out, p1, p2, COLOR_BONE if good else COLOR_BONE_LOW, 2 * scale, cv2.LINE_AA)
    highlight = set(SIDE_LANDMARKS[side].values()) if side else set()
    for j in range(len(xy)):
        if not np.isfinite(xy[j]).all():
            continue
        p = tuple(int(round(v)) for v in xy[j])
        good = visibility[j] >= min_vis
        if j in highlight:
            cv2.circle(out, p, 5 * scale, COLOR_JOINT if good else COLOR_JOINT_LOW, -1, cv2.LINE_AA)
        else:
            cv2.circle(out, p, 3 * scale, COLOR_BONE if good else COLOR_BONE_LOW, -1, cv2.LINE_AA)
    return out


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Semi-transparent text box in the top-left corner."""
    scale = max(0.5, max(frame.shape[:2]) / 1100)
    thickness = max(1, int(round(scale * 2)))
    line_h = int(32 * scale)
    width = int(max(len(s) for s in lines) * 15 * scale) + 20
    height = line_h * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for k, text in enumerate(lines):
        y = 8 + line_h * (k + 1)
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, COLOR_TEXT, thickness, cv2.LINE_AA)
    return frame


def annotate_frame(
    frame: np.ndarray,
    i: int,
    seq: LandmarkSequence,
    fm: FrameMetrics,
    reps_done: int,
    cfg: AnalysisConfig,
) -> np.ndarray:
    draw_pose(frame, seq.image_xy[i], seq.visibility[i], fm.side, cfg.pose.min_landmark_visibility)
    knee = fm.smoothed["knee_angle"][i]
    lean = fm.smoothed["torso_lean"][i]
    lines = [f"Reps: {reps_done}"]
    lines.append(f"Knee: {knee:.0f} deg" if np.isfinite(knee) else "Knee: not visible")
    lines.append(f"Torso lean: {lean:.0f} deg" if np.isfinite(lean) else "Torso lean: not visible")
    if not seq.detected[i]:
        lines.append("No person detected")
    draw_hud(frame, lines)
    return frame


def render_overlay(
    video_path: str | Path,
    seq: LandmarkSequence,
    fm: FrameMetrics,
    reps: RepDetectionResult,
    cfg: AnalysisConfig,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[bytes | None, dict[int, bytes]]:
    """Write the annotated WebM and grab a PNG at each rep's lowest point.

    Returns (webm_bytes or None if encoding failed, {rep_number: png_bytes}).
    Temporary files are deleted before returning.
    """
    bottoms = {r.bottom_index: r.rep_number for r in reps.reps}
    ends = sorted(r.end_index for r in reps.reps)
    key_frames: dict[int, bytes] = {}
    fd, tmp_path = tempfile.mkstemp(suffix=".webm", prefix="ratemyform_overlay_")
    os.close(fd)
    writer = None
    try:
        for k, _src, _t, frame in iter_frames(video_path, cfg.pose):
            if k >= seq.num_frames:
                break
            reps_done = sum(1 for e in ends if e <= k)
            annotate_frame(frame, k, seq, fm, reps_done, cfg)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"VP80"), float(seq.fps), (w, h))
                if not writer.isOpened():
                    writer = None
                    break
            writer.write(frame)
            if k in bottoms:
                ok, png = cv2.imencode(".png", frame)
                if ok:
                    key_frames[bottoms[k]] = png.tobytes()
            if progress and k % 10 == 0:
                progress(min(0.99, (k + 1) / max(1, seq.num_frames)), "Drawing skeleton overlay")
        if writer is not None:
            writer.release()
            writer = None
            data = Path(tmp_path).read_bytes()
            return (data if data else None), key_frames
        return None, key_frames
    finally:
        if writer is not None:
            writer.release()
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def render_synthetic_frame(seq: LandmarkSequence, fm: FrameMetrics, i: int, reps_done: int, cfg: AnalysisConfig) -> bytes:
    """Draw a synthetic skeleton on a blank canvas (no real video exists).

    Used for the labeled synthetic demo so the results page can still show a
    skeleton picture. The canvas is clearly marked SYNTHETIC.
    """
    frame = np.full((seq.height, seq.width, 3), 40, dtype=np.uint8)
    annotate_frame(frame, i, seq, fm, reps_done, cfg)
    h = frame.shape[0]
    cv2.putText(
        frame,
        "SYNTHETIC DATA - not a real person",
        (18, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.6, h / 1400),
        (80, 180, 255),
        2,
        cv2.LINE_AA,
    )
    ok, png = cv2.imencode(".png", frame)
    return png.tobytes() if ok else b""
