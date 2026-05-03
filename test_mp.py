import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
# Use model_selection=1 for full-range (distance + multiple faces)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

frame = np.zeros((720, 1280, 3), dtype=np.uint8)
results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
print("Mediapipe works. Results:", results.detections)
