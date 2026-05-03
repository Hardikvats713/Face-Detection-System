import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
_face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_faces(frame, conf_threshold: float = 0.5):
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _face_detector.process(rgb_frame)

    faces = []
    if results.detections:
        for detection in results.detections:
            if detection.score[0] < conf_threshold:
                continue

            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            x2 = x1 + width
            y2 = y1 + height

            pad_w = int(width * 0.15)
            pad_h = int(height * 0.15)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, int(y1 - pad_h * 1.5))  # more padding on top for forehead
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)

            if x2 <= x1 or y2 <= y1:
                continue

            faces.append((x1, y1, x2, y2))

    return faces

print("Compiled fine!")
