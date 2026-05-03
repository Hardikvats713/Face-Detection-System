import cv2
import numpy as np
import mediapipe as mp
import logging
import os

log = logging.getLogger(__name__)

# ── MediaPipe Detectors ────────────────────────────────────────────────────────
# model_selection=0: Short-range (≤2m), more accurate for close webcam faces
# model_selection=1: Full-range (≤5m), better for far/small faces
mp_face_detection = mp.solutions.face_detection
_mp_short = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
_mp_full  = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)

# ── OpenCV DNN SSD Fallback ────────────────────────────────────────────────────
# This detector is much more robust to head tilts, looking down, eyes down, etc.
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prototxt = os.path.join(_base_dir, "deploy.prototxt")
_caffemodel = os.path.join(_base_dir, "res10_300x300_ssd_iter_140000.caffemodel")

_ssd_net = None
if os.path.exists(_prototxt) and os.path.exists(_caffemodel):
    _ssd_net = cv2.dnn.readNetFromCaffe(_prototxt, _caffemodel)
    log.info("OpenCV DNN SSD face detector loaded (fallback for angled faces).")
else:
    log.warning("SSD model files not found — fallback detector disabled.")


def _pad_box(x1, y1, x2, y2, w, h):
    """Add generous padding around a tight face box for better FaceNet embeddings."""
    width = x2 - x1
    height = y2 - y1
    pad_w = int(width * 0.30)
    pad_h = int(height * 0.30)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, int(y1 - pad_h * 1.2))
    x2 = min(w, x2 + pad_w)
    y2 = min(h, int(y2 + pad_h * 1.3))

    return x1, y1, x2, y2


def _detect_mediapipe(rgb_frame, h, w, detector, conf_threshold):
    """Run MediaPipe face detection and return padded bounding boxes."""
    results = detector.process(rgb_frame)
    faces = []
    if results.detections:
        for detection in results.detections:
            if detection.score[0] < conf_threshold:
                continue

            bbox = detection.location_data.relative_bounding_box
            bx1 = int(bbox.xmin * w)
            by1 = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            bx2 = bx1 + bw
            by2 = by1 + bh

            x1, y1, x2, y2 = _pad_box(bx1, by1, bx2, by2, w, h)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))
    return faces


def _detect_ssd(frame, h, w, conf_threshold):
    """Run OpenCV DNN SSD face detection — robust to head angles & looking down."""
    if _ssd_net is None:
        return []

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    _ssd_net.setInput(blob)
    detections = _ssd_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue

        bx1 = int(detections[0, 0, i, 3] * w)
        by1 = int(detections[0, 0, i, 4] * h)
        bx2 = int(detections[0, 0, i, 5] * w)
        by2 = int(detections[0, 0, i, 6] * h)

        # Clamp to frame bounds
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(w, bx2)
        by2 = min(h, by2)

        if bx2 <= bx1 or by2 <= by1:
            continue

        # SSD boxes are already fairly generous, but add a little padding
        width = bx2 - bx1
        height = by2 - by1
        pad_w = int(width * 0.15)
        pad_h = int(height * 0.15)

        x1 = max(0, bx1 - pad_w)
        y1 = max(0, by1 - pad_h)
        x2 = min(w, bx2 + pad_w)
        y2 = min(h, by2 + int(pad_h * 1.2))

        if x2 > x1 and y2 > y1:
            faces.append((x1, y1, x2, y2))

    return faces


def detect_faces(frame, conf_threshold: float = 0.3, use_fallbacks: bool = False):
    """
    Detects faces — optimized for speed.
    
    By default only uses MediaPipe short-range (fastest, covers 95% webcam cases).
    Set use_fallbacks=True to cascade through full-range + SSD.
    """
    if frame is None or frame.size == 0:
        raise ValueError("detect_faces() received an empty or None frame.")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Primary: MediaPipe short-range (fastest for webcam)
    faces = _detect_mediapipe(rgb_frame, h, w, _mp_short, conf_threshold)
    if faces:
        return faces

    if not use_fallbacks:
        return []

    # Fallback 1: MediaPipe full-range
    faces = _detect_mediapipe(rgb_frame, h, w, _mp_full, conf_threshold)
    if faces:
        log.info(f"MediaPipe (full-range) found {len(faces)} face(s).")
        return faces

    # Fallback 2: SSD
    faces = _detect_ssd(frame, h, w, conf_threshold)
    if faces:
        log.info(f"SSD fallback found {len(faces)} face(s).")
        return faces

    return []
