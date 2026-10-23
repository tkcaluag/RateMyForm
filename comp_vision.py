import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'pose_landmarker_full.task'

# Drawing Utilities - list of tuples that represent the lines between different pose points
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# Post Estimation Model
options = vision.PoseLandmarkerOptions(
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH), # Uses the task file downloaded for the model that contains the trained neural network to find body landmarks
    running_mode = vision.RunningMode.VIDEO, # tells the detector what kind of input to expect (IMAGE, VIDEO, LIVE_STREAM)
    num_poses = 1 # Maximum number of people to detect
)

pose = vision.PoseLandmarker.create_from_options(options)

# Accessing camera and outputting camera feed

capture = cv2.VideoCapture(0) # If you get an error, may have to change parameter. The number inside represents your camera.
timestamp_ms = 0
while capture.isOpened():
    ret, frame = capture.read() # frame is the image from camera
    cv2.imshow('Mediapipe Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'): # Checks for key press and if the key pressed is 'q' to close windows
        break

capture.release()
cv2.destroyAllWindows()

