"""Integration tests for the real MediaPipe + OpenCV layer.

These use generated videos with NO person in them (we have no licensed real
footage in the repo). They verify the plumbing: model loading, VIDEO-mode
timestamps, frame sampling, landmark index constants, safe failure when no
person is found, overlay encoding, and temp-file cleanup. They do NOT verify
detection accuracy on real people; that requires real recordings (see
docs/TESTING.md).
"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from ratemyform import landmarks as L
from ratemyform.config import PoseConfig
from ratemyform.pose_detection import (
    VideoReadError,
    detect_image,
    extract_landmarks,
    frame_stride,
    model_path,
    probe_video,
)

pytestmark = pytest.mark.integration
requires_model = pytest.mark.skipif(
    not model_path("full").is_file(),
    reason="MediaPipe model not downloaded: run python scripts/download_model.py",
)


def make_video(path: Path, fps=30.0, seconds=2.0, size=(640, 360)) -> Path:
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(fps * seconds)):
        frame = np.full((h, w, 3), 80, dtype=np.uint8)
        cv2.rectangle(frame, (20 + i % (w - 60), 100), (60 + i % (w - 60), 200), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


def test_landmark_constants_match_mediapipe():
    from mediapipe.tasks.python import vision

    P = vision.PoseLandmark
    assert L.LEFT_SHOULDER == P.LEFT_SHOULDER and L.RIGHT_SHOULDER == P.RIGHT_SHOULDER
    assert L.LEFT_HIP == P.LEFT_HIP and L.RIGHT_HIP == P.RIGHT_HIP
    assert L.LEFT_KNEE == P.LEFT_KNEE and L.RIGHT_KNEE == P.RIGHT_KNEE
    assert L.LEFT_ANKLE == P.LEFT_ANKLE and L.RIGHT_ANKLE == P.RIGHT_ANKLE
    assert L.LEFT_HEEL == P.LEFT_HEEL and L.RIGHT_FOOT_INDEX == P.RIGHT_FOOT_INDEX
    assert L.LEFT_EAR == P.LEFT_EAR and L.NOSE == P.NOSE
    assert len(list(P)) == L.NUM_LANDMARKS == 33


def test_skeleton_connections_match_mediapipe():
    from mediapipe.tasks.python import vision

    theirs = {tuple(sorted((c.start, c.end))) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS}
    ours = {tuple(sorted(c)) for c in L.POSE_CONNECTIONS}
    assert ours == theirs


@pytest.mark.parametrize("src, expected", [(24, 1), (30, 1), (30.0, 1), (50, 2), (59.94, 2), (60, 2), (120, 4), (240, 8)])
def test_frame_stride(src, expected):
    assert frame_stride(src, 30.0) == expected


def test_probe_and_corrupt_file(tmp_path):
    info = probe_video(make_video(tmp_path / "ok.mp4", seconds=1.0))
    assert info.width == 640 and info.height == 360
    assert info.fps == pytest.approx(30.0)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(VideoReadError):
        probe_video(bad)


@requires_model
def test_blank_image_has_no_detection():
    seq = detect_image(np.full((480, 640, 3), 127, dtype=np.uint8))
    assert seq.num_frames == 1 and not seq.detected.any()


@requires_model
def test_video_mode_runs_and_subsamples_high_fps(tmp_path):
    path = make_video(tmp_path / "v60.mp4", fps=60.0, seconds=2.0)
    seq = extract_landmarks(path, PoseConfig(max_analysis_fps=30.0))
    assert seq.fps == pytest.approx(30.0)
    assert 55 <= seq.num_frames <= 61
    assert np.all(np.diff(seq.timestamps_s) > 0)
    assert seq.metadata["frame_stride"] == 2
    assert not seq.detected.any()  # no person in the generated video
    assert np.isnan(seq.image_xy).all()


@requires_model
def test_analyze_video_without_person_fails_safely(tmp_path, average_body):
    from ratemyform.pipeline import analyze_video

    path = make_video(tmp_path / "blank.mp4", seconds=2.0)
    before = set(os.listdir(tempfile.gettempdir()))
    res = analyze_video(path, average_body)
    after = set(os.listdir(tempfile.gettempdir()))
    assert res.rep_count == 0 and res.overall_personalized is None
    assert any("detected in only 0%" in w for w in res.warnings)
    # Overlay is a real WebM (EBML magic bytes) and temp overlay files were removed.
    assert res.overlay_video is not None and res.overlay_video[:4] == b"\x1a\x45\xdf\xa3"
    assert not {f for f in after - before if f.startswith("ratemyform_overlay_")}
