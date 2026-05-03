import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1

log = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# ── Face alignment using MediaPipe landmarks ──────────────────────────────────
_mp_face_mesh = mp.solutions.face_mesh
_aligner = _mp_face_mesh.FaceMesh(
    static_image_mode=False,   # PERF: video mode is much faster
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

# MediaPipe landmark indices for eye centers
LEFT_EYE_IDX = 468    # left eye center (if refined) or approximate
RIGHT_EYE_IDX = 473   # right eye center (if refined) or approximate
# Fallback: average of eye corner landmarks
LEFT_EYE_CORNERS = [33, 133]    # left eye inner/outer corners
RIGHT_EYE_CORNERS = [362, 263]  # right eye inner/outer corners

# Minimum face crop size
MIN_FACE_SIZE = 40  # pixels


def _align_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Align a face crop so the eyes are horizontal.
    This makes embeddings more consistent across head tilts/angles.
    
    If alignment fails (no landmarks found), returns the original crop.
    """
    try:
        h, w = face_bgr.shape[:2]
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        results = _aligner.process(rgb)

        if not results.multi_face_landmarks:
            return face_bgr  # Can't align, use as-is

        lm = results.multi_face_landmarks[0].landmark

        # Get eye center positions (average of inner/outer corners)
        left_eye = np.array([
            (lm[LEFT_EYE_CORNERS[0]].x * w + lm[LEFT_EYE_CORNERS[1]].x * w) / 2,
            (lm[LEFT_EYE_CORNERS[0]].y * h + lm[LEFT_EYE_CORNERS[1]].y * h) / 2,
        ])
        right_eye = np.array([
            (lm[RIGHT_EYE_CORNERS[0]].x * w + lm[RIGHT_EYE_CORNERS[1]].x * w) / 2,
            (lm[RIGHT_EYE_CORNERS[0]].y * h + lm[RIGHT_EYE_CORNERS[1]].y * h) / 2,
        ])

        # Calculate rotation angle to make eyes horizontal
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # Only rotate if the angle is significant (> 2 degrees)
        if abs(angle) < 2.0:
            return face_bgr

        # Rotate around the midpoint of the eyes
        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(face_bgr, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE)
        return aligned

    except Exception as e:
        log.debug(f"Face alignment failed: {e}")
        return face_bgr


def get_embedding(face: np.ndarray) -> np.ndarray:
    """
    Generate a 512-d face embedding from a BGR face crop.
    
    Pipeline: validate → align → normalize → resize → embed → L2-normalize
    """
    # Guard against None or empty input
    if face is None or face.size == 0:
        raise ValueError("get_embedding() received an empty or None face crop.")

    # Check minimum face size
    h, w = face.shape[:2]
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        raise ValueError(
            f"Face crop too small ({w}x{h}). Minimum is {MIN_FACE_SIZE}x{MIN_FACE_SIZE}px."
        )

    # Ensure face is uint8
    if face.dtype != np.uint8:
        face = np.clip(face * 255 if face.max() <= 1.0 else face, 0, 255).astype(np.uint8)

    # Ensure 3-channel BGR image
    if len(face.shape) == 2:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    elif face.shape[2] == 4:
        face = cv2.cvtColor(face, cv2.COLOR_BGRA2BGR)

    # ── Face alignment — normalize head tilt ──────────────────────────────
    face = _align_face(face)

    # Convert BGR → RGB (FaceNet was trained on RGB)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize to 160x160 using LANCZOS for best quality
    face_resized = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)

    # Normalize to [-1, 1] as expected by FaceNet/VGGFace2
    face_tensor = (torch.tensor(face_resized, dtype=torch.float32) / 127.5) - 1.0

    # shape: (H, W, C) → (1, C, H, W)
    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(face_tensor)

    # L2-normalize the embedding
    emb = F.normalize(emb, p=2, dim=1)

    return emb.cpu().numpy()[0]