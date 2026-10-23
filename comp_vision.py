import cv2
import mediapipe as mp
import numpy as np

# Drawing Utilities
mp_drawing = mp.solutions.drawing_utils

# Post Estimation Models
mp_pose = mp.solutions.pose


# Accessing camera and outputting camera feed

capture = cv2.VideoCapture(0) # If you get an error, may have to change parameter. The number inside represents your camera.
while capture.isOpened():
    ret, frame = capture.read() # frame is the image from camera
    cv2.imshow('Mediapipe Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'): # Checks for key press and if the key pressed is 'q' to close windows
        break

capture.release()
cv2.destroyAllWindows()

