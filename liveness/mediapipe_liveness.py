import mediapipe as mp
import numpy as np
import cv2
import logging
import time
from collections import deque

log = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Key landmark indices for motion tracking (nose tip, chin, forehead, eye corners)
MOTION_LANDMARKS = [1, 4, 10, 152, 33, 263, 234, 454]

# ── Thresholds ─────────────────────────────────────────────────────────────────
MIN_Z_RANGE = 0.05
MIN_TEXTURE_VARIANCE = 12.0
MIN_EDGE_DENSITY = 0.01

# Blink detection
BLINK_EAR_THRESHOLD = 0.21   # EAR below this = eyes closed
MIN_BLINK_FRAMES = 1         # At least 1 blink needed

# Motion detection — real faces have natural micro-sway
# Standard deviation of landmark positions over a sliding window
MIN_MOTION_STD = 0.003       # Minimum position variance (real faces ~0.005-0.02)
LIVENESS_WINDOW = 8          # Number of frames to track
REQUIRED_EVIDENCE_FRAMES = 4 # Frames needed before granting access


# ── Temporal Liveness Tracker ──────────────────────────────────────────────────
class LivenessTracker:
    """
    Tracks per-person liveness evidence across multiple frames.
    
    A photo held up to the camera will:
    - Never blink (EAR stays constant)
    - Have near-zero landmark motion (perfectly still)
    
    A real person will:
    - Blink naturally every few seconds
    - Show micro-movements (head sway, breathing)
    """

    def __init__(self):
        self.sessions = {}   # name → session data
        self.lock_timeout = 15.0  # seconds before session expires

    def _new_session(self):
        return {
            'ear_history': deque(maxlen=LIVENESS_WINDOW),
            'landmark_history': deque(maxlen=LIVENESS_WINDOW),
            'blink_count': 0,
            'was_eye_closed': False,
            'frames_seen': 0,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'granted': False,
            'granted_at': 0,
        }

    def cleanup(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [k for k, v in self.sessions.items()
                   if now - v['last_seen'] > self.lock_timeout]
        for k in expired:
            del self.sessions[k]

    def update(self, name, ear, landmarks_xy):
        """
        Feed a new frame's data for a person.
        Returns (is_live, reason, details)
        """
        self.cleanup()

        if name not in self.sessions:
            self.sessions[name] = self._new_session()

        s = self.sessions[name]
        s['last_seen'] = time.time()
        s['frames_seen'] += 1

        # ── Track EAR for blink detection ──────────────────────────────────
        if ear is not None:
            s['ear_history'].append(ear)

            # Blink = EAR drops below threshold then recovers
            if ear < BLINK_EAR_THRESHOLD:
                if not s['was_eye_closed']:
                    s['was_eye_closed'] = True
            else:
                if s['was_eye_closed']:
                    s['blink_count'] += 1
                    log.info(f"BLINK detected for {name} (total: {s['blink_count']})")
                    s['was_eye_closed'] = False

        # ── Track landmark positions for motion detection ──────────────────
        if landmarks_xy is not None:
            s['landmark_history'].append(landmarks_xy)

        # ── Evaluate liveness ──────────────────────────────────────────────
        details = {
            'frames_seen': s['frames_seen'],
            'blink_count': s['blink_count'],
        }

        # Need enough frames before we can judge
        if s['frames_seen'] < REQUIRED_EVIDENCE_FRAMES:
            details['motion_std'] = None
            return None, f"Analyzing... ({s['frames_seen']}/{REQUIRED_EVIDENCE_FRAMES} frames)", details

        # Already granted recently — stay granted for 10 seconds
        if s['granted'] and (time.time() - s['granted_at'] < 10.0):
            return True, "Live (verified).", details

        # ── Check 1: Motion variance ───────────────────────────────────────
        motion_std = 0.0
        if len(s['landmark_history']) >= 3:
            positions = np.array(list(s['landmark_history']))  # (N, K, 2)
            # Std of each landmark's x,y across frames
            per_landmark_std = np.std(positions, axis=0)  # (K, 2)
            motion_std = float(np.mean(per_landmark_std))

        details['motion_std'] = round(motion_std, 5)

        # ── Check 2: EAR variance (eyes should fluctuate naturally) ────────
        ear_std = 0.0
        if len(s['ear_history']) >= 3:
            ear_std = float(np.std(list(s['ear_history'])))
        details['ear_std'] = round(ear_std, 5)

        # ── Decision ───────────────────────────────────────────────────────
        has_blink = s['blink_count'] >= MIN_BLINK_FRAMES
        has_motion = motion_std >= MIN_MOTION_STD

        if has_blink or has_motion:
            s['granted'] = True
            s['granted_at'] = time.time()
            log.info(f"TEMPORAL LIVENESS PASS: {name} (blinks={s['blink_count']}, motion={motion_std:.5f})")
            return True, "Live face.", details
        else:
            reason = "Hold still... verifying you're real. "
            if not has_motion:
                reason += f"No movement detected (std={motion_std:.5f}). "
            if not has_blink:
                reason += f"No blink detected. "
            return False, reason.strip(), details

    def reset(self, name=None):
        if name:
            self.sessions.pop(name, None)
        else:
            self.sessions.clear()


# Global tracker instance
liveness_tracker = LivenessTracker()


def _eye_aspect_ratio(landmarks, indices, w, h):
    pts = np.array([
        (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
        for idx in indices
    ], dtype=np.float32)

    vertical_1  = np.linalg.norm(pts[1] - pts[5])
    vertical_2  = np.linalg.norm(pts[2] - pts[4])
    horizontal  = np.linalg.norm(pts[0] - pts[3])

    if horizontal == 0.0:
        return 1.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def _texture_score(face_crop):
    """
    Analyze texture to detect printed photos or screen displays.
    Returns (laplacian_variance, edge_density)
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0, 0.0

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(laplacian.var())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    return lap_var, edge_density


def _color_distribution_check(face_crop):
    """
    Check color channel distribution — screens/photos have different
    color properties than real skin under natural lighting.
    Returns True if color distribution looks like a real face.
    """
    if face_crop is None or face_crop.size == 0:
        return True

    ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
    cr_std = float(np.std(ycrcb[:, :, 1]))
    cb_std = float(np.std(ycrcb[:, :, 2]))
    return cr_std > 3.0 or cb_std > 3.0


def get_liveness_metrics(frame: np.ndarray):
    """Returns (ear, z_range, head_pose, landmarks_xy) or (None, None, 'center', None)"""
    if frame is None or frame.size == 0:
        return None, None, "center", None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None, "center", None

    h, w = frame.shape[:2]
    face_landmarks = results.multi_face_landmarks[0]

    # 1. EAR for Blink
    left_ear  = _eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE,  w, h)
    right_ear = _eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
    ear = (left_ear + right_ear) / 2.0

    # 2. Depth Check (Z-range)
    z_coords = [lm.z for lm in face_landmarks.landmark]
    z_range = max(z_coords) - min(z_coords)

    # 3. Head Pose Estimation
    nose = face_landmarks.landmark[1]
    left_edge = face_landmarks.landmark[234]
    right_edge = face_landmarks.landmark[454]
    top = face_landmarks.landmark[10]
    bottom = face_landmarks.landmark[152]

    yaw_ratio = abs(nose.x - left_edge.x) / (abs(right_edge.x - nose.x) + 1e-6)
    pitch_ratio = abs(nose.y - top.y) / (abs(bottom.y - nose.y) + 1e-6)

    if yaw_ratio > 1.25:
        pose = "right"
    elif yaw_ratio < 0.8:
        pose = "left"
    elif pitch_ratio < 0.85:
        pose = "up"
    elif pitch_ratio > 1.25:
        pose = "down"
    else:
        pose = "center"

    # 4. Extract key landmark positions for motion tracking
    landmarks_xy = np.array([
        (face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
        for idx in MOTION_LANDMARKS
    ], dtype=np.float32)

    return ear, z_range, pose, landmarks_xy


def check_liveness(face_crop, full_frame=None, face_box=None, person_name=None):
    """
    Two-stage liveness check:
      Stage 1 (per-frame): Z-depth, texture, color — catches obvious fakes
      Stage 2 (temporal):  Blink + micro-motion over multiple frames — catches photos
    
    Returns:
        (is_live: bool or None, reason: str, details: dict)
        None = still verifying (need more frames)
    """
    details = {}

    # ── Stage 1: Per-frame static checks ──────────────────────────────────

    # 1a. Face Mesh depth & pose check
    ear, z_range, pose, landmarks_xy = get_liveness_metrics(face_crop)

    details['ear'] = round(ear, 4) if ear is not None else None
    details['z_range'] = round(z_range, 4) if z_range is not None else None
    details['pose'] = pose

    if z_range is not None and z_range < MIN_Z_RANGE:
        log.info(f"LIVENESS FAIL: z_range={z_range:.4f} < {MIN_Z_RANGE} (flat)")
        return False, "Flat surface detected (photo/screen).", details

    # 1b. Texture analysis
    lap_var, edge_density = _texture_score(face_crop)
    details['texture_variance'] = round(lap_var, 2)
    details['edge_density'] = round(edge_density, 4)

    if lap_var < MIN_TEXTURE_VARIANCE and lap_var > 0:
        log.info(f"LIVENESS FAIL: texture_var={lap_var:.2f} < {MIN_TEXTURE_VARIANCE}")
        return False, "Photo detected (low texture).", details

    # 1c. Color distribution
    color_ok = _color_distribution_check(face_crop)
    details['color_natural'] = color_ok

    if not color_ok:
        log.info("LIVENESS FAIL: unnatural color distribution")
        return False, "Photo detected (unnatural colors).", details

    # ── Stage 2: Temporal liveness (blink + motion) ───────────────────────
    if person_name:
        is_live, reason, temporal_details = liveness_tracker.update(
            person_name, ear, landmarks_xy
        )
        details.update(temporal_details)

        if is_live is None:
            # Still collecting frames
            return None, reason, details
        elif not is_live:
            return False, reason, details
        else:
            return True, reason, details

    # No person_name provided — fall back to static-only (less secure)
    log.debug(f"LIVENESS OK (static only): z_range={z_range}, texture={lap_var:.1f}")
    return True, "Live face (static check only).", details


def reset_liveness(name=None):
    """Reset liveness state."""
    liveness_tracker.reset(name)
