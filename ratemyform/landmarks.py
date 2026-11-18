"""Landmark indices and the LandmarkSequence container.

MediaPipe Pose returns 33 landmarks per frame. We store a whole video's
landmarks as NumPy arrays (one row per analyzed frame) so later steps can do
vectorized math instead of looping over Python objects.

Three coordinate systems appear in this project. Keeping them separate is
important when explaining results:

1. Normalized image coordinates (MediaPipe `pose_landmarks`): x and y in
   [0, 1] relative to frame width and height. We convert these to PIXELS
   before measuring angles, otherwise a non-square video would distort them.
2. World coordinates (MediaPipe `pose_world_landmarks`): estimated 3D
   positions in meters with the origin at the midpoint of the hips. The model
   card notes the depth (z) is estimated from synthetic data and "not
   metric", so these are estimates, not physical measurements.
3. Real-world body measurements: the tape-measured lengths the user types in
   (centimeters). The video never measures these directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NUM_LANDMARKS = 33

# MediaPipe Pose landmark indices (verified against
# mediapipe.tasks.python.vision.PoseLandmark in tests/test_pose_detection.py).
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

SIDE_LANDMARKS = {
    "left": {
        "ear": LEFT_EAR,
        "shoulder": LEFT_SHOULDER,
        "elbow": LEFT_ELBOW,
        "wrist": LEFT_WRIST,
        "hip": LEFT_HIP,
        "knee": LEFT_KNEE,
        "ankle": LEFT_ANKLE,
        "heel": LEFT_HEEL,
        "foot_index": LEFT_FOOT_INDEX,
    },
    "right": {
        "ear": RIGHT_EAR,
        "shoulder": RIGHT_SHOULDER,
        "elbow": RIGHT_ELBOW,
        "wrist": RIGHT_WRIST,
        "hip": RIGHT_HIP,
        "knee": RIGHT_KNEE,
        "ankle": RIGHT_ANKLE,
        "heel": RIGHT_HEEL,
        "foot_index": RIGHT_FOOT_INDEX,
    },
}

# Skeleton connections used for drawing (same pairs as MediaPipe's
# PoseLandmarksConnections.POSE_LANDMARKS; checked in tests).
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


@dataclass
class LandmarkSequence:
    """Pose landmarks for every analyzed frame of one video.

    Shapes (N = number of analyzed frames):
        timestamps_s   (N,)       seconds from the start of the video
        frame_indices  (N,)       index of the frame in the original video
        detected       (N,)       True if MediaPipe found a person
        image_xy       (N, 33, 2) pixel coordinates; NaN when not detected
        visibility     (N, 33)    0..1 probability landmark is visible
        world_xyz      (N, 33, 3) meters, hip-centered; NaN when not detected
    """

    timestamps_s: np.ndarray
    frame_indices: np.ndarray
    detected: np.ndarray
    image_xy: np.ndarray
    visibility: np.ndarray
    world_xyz: np.ndarray
    width: int
    height: int
    fps: float
    source: str = "video"  # "video" or "synthetic"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.timestamps_s)
        expected = {
            "frame_indices": (n,),
            "detected": (n,),
            "image_xy": (n, NUM_LANDMARKS, 2),
            "visibility": (n, NUM_LANDMARKS),
            "world_xyz": (n, NUM_LANDMARKS, 3),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")
        if n > 1 and np.any(np.diff(self.timestamps_s) <= 0):
            raise ValueError("timestamps_s must be strictly increasing")

    @property
    def num_frames(self) -> int:
        return int(len(self.timestamps_s))

    @property
    def duration_s(self) -> float:
        if self.num_frames < 2:
            return 0.0
        return float(self.timestamps_s[-1] - self.timestamps_s[0])

    @property
    def detection_rate(self) -> float:
        if self.num_frames == 0:
            return 0.0
        return float(np.mean(self.detected))

    @classmethod
    def empty(cls, num_frames: int, width: int, height: int, fps: float, source: str = "video") -> "LandmarkSequence":
        """Allocate arrays for `num_frames` frames with nothing detected."""
        return cls(
            timestamps_s=np.arange(num_frames, dtype=float) / fps,
            frame_indices=np.arange(num_frames, dtype=int),
            detected=np.zeros(num_frames, dtype=bool),
            image_xy=np.full((num_frames, NUM_LANDMARKS, 2), np.nan),
            visibility=np.zeros((num_frames, NUM_LANDMARKS)),
            world_xyz=np.full((num_frames, NUM_LANDMARKS, 3), np.nan),
            width=width,
            height=height,
            fps=fps,
            source=source,
        )
