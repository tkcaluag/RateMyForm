import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_landmarker_full.task')

# Drawing Utilities - list of tuples that represent the lines between different pose points
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def to_array(landmarks):
    return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks], dtype=float)


def draw_landmarks(image, landmarks): # takes in camera image and landmarks which is a list of 33 points for one person

    if not isinstance(landmarks, np.ndarray):
        landmarks = to_array(landmarks)

    h, w = image.shape[:2] # height and width of image

    # Media Pipe returns each landmark's x and y as normalized values between 0 and 1 (ex. x = 0.5 would be halfway across the image)
    # Must multiply x by the width and y by the height then convert them to integers because pixels have to be whole numbers.
    points = [(int(x * w), int(y * h)) for x, y in landmarks[:, :2]]

    # Drawing the lines
    # Loops over every connection tuple into a and b and drawing a line between those two points
    for a, b in POSE_CONNECTIONS:
        cv2.line(image, points[a], points[b], (255, 255, 255), 2) # (255, 255, 255) is the color white and 2 is the line thickness

    # Drawing the dots
    for p in points:
        cv2.circle(image, p, 4, (0, 255, 0), -1) # Radius of 4. OpenCV uses BGR isntead of RGB so (0, 255, 0) is green. A thickness of -1 fills the circle instead of outlining

    return image


# Post Estimation Model
def create_landmarker(running_mode=vision.RunningMode.VIDEO):
    options = vision.PoseLandmarkerOptions(
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH), # Uses the task file downloaded for the model that contains the trained neural network to find body landmarks
        running_mode = running_mode, # tells the detector what kind of input to expect (IMAGE, VIDEO, LIVE_STREAM)
        num_poses = 1 # Maximum number of people to detect
    )
    return vision.PoseLandmarker.create_from_options(options)


def _first_pose(results):
    if not results.pose_landmarks:
        return None, None
    return to_array(results.pose_landmarks[0]), to_array(results.pose_world_landmarks[0])


def detect_image(image_rgb):
    """Runs pose detection on one RGB image (numpy array or PIL image).

    Returns a dict with 'landmarks' (normalized image coords) and 'world_landmarks'
    (meters, origin at the hips), or None if no person was detected.
    """
    image_rgb = np.ascontiguousarray(np.asarray(image_rgb, dtype=np.uint8))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    with create_landmarker(vision.RunningMode.IMAGE) as pose:
        landmarks, world = _first_pose(pose.detect(mp_image))

    if landmarks is None:
        return None
    return {'landmarks': landmarks, 'world_landmarks': world}


def detect_video(video_path, max_frames=None):
    """Runs pose detection on every frame of a video file.

    Returns (frames, fps). Each frame is a dict with 'timestamp_ms', 'landmarks'
    and 'world_landmarks'. The landmark entries are None on frames with no person.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    with create_landmarker(vision.RunningMode.VIDEO) as pose:
        index = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            timestamp_ms = int(index * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarks, world = _first_pose(pose.detect_for_video(mp_image, timestamp_ms))

            frames.append({'timestamp_ms': timestamp_ms, 'landmarks': landmarks, 'world_landmarks': world})

            index += 1
            if max_frames and index >= max_frames:
                break

    capture.release()
    return frames, fps


def write_annotated_video(video_path, frames, output_path):
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for f in frames:
        ret, frame = capture.read()
        if not ret:
            break
        if f['landmarks'] is not None:
            draw_landmarks(frame, f['landmarks'])
        writer.write(frame)

    capture.release()
    writer.release()
    return output_path


def run_webcam(camera_index=0):
    # Accessing camera and outputting camera feed
    capture = cv2.VideoCapture(camera_index) # If you get an error, may have to change parameter. The number inside represents your camera.
    pose = create_landmarker(vision.RunningMode.VIDEO)
    timestamp_ms = 0 # each frame needs a timestamp which is used to track the person across frames

    while capture.isOpened():
        ret, frame = capture.read() # frame is the image from camera
        if not ret:
            break

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


if __name__ == '__main__':
    run_webcam()
