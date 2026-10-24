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

def draw_landmarks(image, landmarks): # takes in camera image and landmarks which is a list of 33 points for one person

    h, w = image.shape[:2] # height and width of image

    # Media Pipe returns each landmark's x and y as normalized values between 0 and 1 (ex. x = 0.5 would be halfway across the image)
    # Must multiply x by the width and y by the height then convert them to integers because pixels have to be whole numbers.
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Drawing the lines
    # Loops over every connection tuple into a and b and drawing a line between those two points
    for a, b in POSE_CONNECTIONS:
        cv2.line(image, points[a], points[b], (255, 255, 255), 2) # (255, 255, 255) is the color white and 2 is the line thickness

    # Drawing the dots
    for p in points:
        cv2.circle(image, p, 4, (0, 255, 0), -1) # Radius of 4. OpenCV uses BGR isntead of RGB so (0, 255, 0) is green. A thickness of -1 fills the circle instead of outlining

# Post Estimation Model
options = vision.PoseLandmarkerOptions(
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH), # Uses the task file downloaded for the model that contains the trained neural network to find body landmarks
    running_mode = vision.RunningMode.VIDEO, # tells the detector what kind of input to expect (IMAGE, VIDEO, LIVE_STREAM)
    num_poses = 1 # Maximum number of people to detect
)

pose = vision.PoseLandmarker.create_from_options(options)

# Accessing camera and outputting camera feed

capture = cv2.VideoCapture(0) # If you get an error, may have to change parameter. The number inside represents your camera.
timestamp_ms = 0 # each frame needs a timestamp which is used to track the person across frames
while capture.isOpened():
    ret, frame = capture.read() # frame is the image from camera

    # Run pose detection

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converts from BGR to RGB (MediaPipe uses RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms += 10
    results = pose.detect_for_video(mp_image, timestamp_ms) # returns body landmarks detected

    # Draw each detected person's landmarks onto the frame
    for landmarks in results.pose_landmarks:
        draw_landmarks(frame, landmarks)

    cv2.imshow('Mediapipe Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'): # Checks for key press and if the key pressed is 'q' to close windows
        break

capture.release()
pose.close()
cv2.destroyAllWindows()

