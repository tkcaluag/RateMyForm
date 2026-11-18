"""Video reading (OpenCV) and pose landmark detection (MediaPipe).

Status Report 2 describes Kyle restructuring the original webcam script into
importable functions that run pose detection on an uploaded video and return
the 33 landmarks per frame. The team's code was not available when this
module was written, so this is a fresh implementation of that interface.
When the team repository is merged, compare it with Kyle's version and keep
whichever is better tested (see docs/AUDIT.md).

MediaPipe version note: MediaPipe >= 0.10.30 (including 1.x, which is what
installs on Python 3.13) removed the old `mp.solutions.pose` API. This module
uses the current Tasks API: `PoseLandmarker` with a downloaded `.task` model
file (run `python scripts/download_model.py`).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .config import PoseConfig
from .landmarks import NUM_LANDMARKS, LandmarkSequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

ProgressCallback = Callable[[float, str], None]


class ModelNotFoundError(FileNotFoundError):
    """The MediaPipe .task model file has not been downloaded yet."""


class VideoReadError(RuntimeError):
    """OpenCV could not open or decode the video."""


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_s": round(self.duration_s, 2),
        }


def model_path(variant: str = "full") -> Path:
    return MODEL_DIR / f"pose_landmarker_{variant}.task"


def require_model(variant: str = "full") -> Path:
    path = model_path(variant)
    if not path.is_file():
        raise ModelNotFoundError(
            f"MediaPipe model not found at {path}. Run: python scripts/download_model.py --variant {variant}"
        )
    return path


def probe_video(path: str | Path) -> VideoInfo:
    """Read basic properties and confirm at least one frame decodes."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError("The video could not be opened. It may be corrupted or use an unsupported codec.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoReadError("The video opened but no frames could be decoded.")
        height, width = frame.shape[:2]  # trust the decoded frame (handles rotation)
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0:
        raise VideoReadError("The video does not report a valid frame rate.")
    duration = frame_count / fps if frame_count > 0 else 0.0
    return VideoInfo(fps=fps, frame_count=frame_count, width=width, height=height, duration_s=duration)


def frame_stride(source_fps: float, max_fps: float) -> int:
    """Analyze every k-th frame so the analysis rate stays <= max_fps."""
    if source_fps <= max_fps:
        return 1
    return max(1, math.ceil(source_fps / max_fps - 0.01))


def iter_frames(path: str | Path, cfg: PoseConfig) -> Iterator[tuple[int, int, float, np.ndarray]]:
    """Yield (analysis_index, source_frame_index, timestamp_s, frame_bgr).

    Frames are subsampled to <= cfg.max_analysis_fps and downscaled to
    <= cfg.max_frame_long_side_px. Timestamps assume a constant frame rate.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError("The video could not be opened.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise VideoReadError("The video does not report a valid frame rate.")
    stride = frame_stride(fps, cfg.max_analysis_fps)
    src_index = 0
    out_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if src_index % stride == 0:
                h, w = frame.shape[:2]
                scale = cfg.max_frame_long_side_px / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
                yield out_index, src_index, src_index / fps, frame
                out_index += 1
            src_index += 1
    finally:
        cap.release()


def _create_landmarker(cfg: PoseConfig, running_mode: str):
    # Imported here so the rest of the package (and the synthetic tests) work
    # even on machines where MediaPipe is not installed.
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(require_model(cfg.model_variant))),
        running_mode=getattr(vision.RunningMode, running_mode),
        num_poses=1,
        min_pose_detection_confidence=cfg.min_pose_detection_confidence,
        min_pose_presence_confidence=cfg.min_pose_presence_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _to_mp_image(frame_bgr: np.ndarray):
    import mediapipe as mp

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def _result_to_arrays(result, width: int, height: int) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Convert one MediaPipe result to (detected, image_xy_px, visibility, world_xyz)."""
    xy = np.full((NUM_LANDMARKS, 2), np.nan)
    vis = np.zeros(NUM_LANDMARKS)
    world = np.full((NUM_LANDMARKS, 3), np.nan)
    if not result.pose_landmarks or len(result.pose_landmarks[0]) != NUM_LANDMARKS:
        return False, xy, vis, world
    for j, lm in enumerate(result.pose_landmarks[0]):
        xy[j] = (lm.x * width, lm.y * height)  # normalized [0,1] -> pixels
        vis[j] = lm.visibility if lm.visibility is not None else 0.0
    if result.pose_world_landmarks and len(result.pose_world_landmarks[0]) == NUM_LANDMARKS:
        for j, lm in enumerate(result.pose_world_landmarks[0]):
            world[j] = (lm.x, lm.y, lm.z)  # meters, origin = midpoint of hips
    return True, xy, vis, world


def _fill_frame(seq: LandmarkSequence, i: int, result, width: int, height: int) -> None:
    """Copy one MediaPipe result into row i of the sequence arrays."""
    detected, xy, vis, world = _result_to_arrays(result, width, height)
    seq.detected[i] = detected
    seq.image_xy[i] = xy
    seq.visibility[i] = vis
    seq.world_xyz[i] = world


def detect_image(frame_bgr: np.ndarray, cfg: PoseConfig | None = None) -> LandmarkSequence:
    """Run pose detection on a single photo. Returns a 1-frame sequence."""
    cfg = cfg or PoseConfig()
    h, w = frame_bgr.shape[:2]
    seq = LandmarkSequence.empty(1, w, h, fps=1.0)
    with _create_landmarker(cfg, "IMAGE") as landmarker:
        result = landmarker.detect(_to_mp_image(frame_bgr))
    _fill_frame(seq, 0, result, w, h)
    return seq


def extract_landmarks(
    path: str | Path,
    cfg: PoseConfig | None = None,
    progress: ProgressCallback | None = None,
) -> LandmarkSequence:
    """Run MediaPipe Pose (VIDEO mode) on every analyzed frame of a video."""
    cfg = cfg or PoseConfig()
    info = probe_video(path)
    stride = frame_stride(info.fps, cfg.max_analysis_fps)
    expected = max(1, info.frame_count // stride) if info.frame_count > 0 else 1

    times: list[float] = []
    src_indices: list[int] = []
    rows: list[tuple[bool, np.ndarray, np.ndarray, np.ndarray]] = []
    width = height = 0
    last_ts_ms = -1
    started = time.perf_counter()
    with _create_landmarker(cfg, "VIDEO") as landmarker:
        for k, src_i, t, frame in iter_frames(path, cfg):
            height, width = frame.shape[:2]
            # VIDEO mode needs integer, monotonically increasing timestamps.
            ts_ms = int(round(t * 1000.0))
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            result = landmarker.detect_for_video(_to_mp_image(frame), ts_ms)
            times.append(ts_ms / 1000.0)
            src_indices.append(src_i)
            rows.append(_result_to_arrays(result, width, height))
            if progress and k % 5 == 0:
                progress(min(0.99, (k + 1) / expected), f"Detecting pose: frame {k + 1} of ~{expected}")

    n = len(times)
    if n == 0:
        raise VideoReadError("No frames could be read from the video.")
    seq = LandmarkSequence(
        timestamps_s=np.asarray(times, dtype=float),
        frame_indices=np.asarray(src_indices, dtype=int),
        detected=np.array([r[0] for r in rows], dtype=bool),
        image_xy=np.stack([r[1] for r in rows]),
        visibility=np.stack([r[2] for r in rows]),
        world_xyz=np.stack([r[3] for r in rows]),
        width=width,
        height=height,
        fps=info.fps / stride,
        source="video",
    )
    seq.metadata = {
        "model_variant": cfg.model_variant,
        "frame_stride": stride,
        "pose_seconds": round(time.perf_counter() - started, 3),
    }
    if progress:
        progress(1.0, "Pose detection finished")
    return seq
