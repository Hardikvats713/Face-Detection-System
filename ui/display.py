import cv2
import numpy as np

# ── Design constants ──────────────────────────────────────────────────────────
_COLORS = {
    "ALLOW": (0, 255, 0),    # green
    "WAIT":  (0, 255, 255),  # yellow
    "DENY":  (0, 0, 255),    # red
}

_LABELS = {
    "ALLOW": "ACCESS GRANTED",
    "WAIT":  "VERIFYING...",
    "DENY":  "ACCESS DENIED",
}

_OVERLAY_ALPHA  = 0.2   # tint strength
_FONT           = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE     = 1.0
_FONT_THICKNESS = 2
_BORDER_THICKNESS = 6   # colored border around frame edge


def draw_status(
    frame: np.ndarray,
    status: str,
    name: str | None = None,
    score: float | None = None,
    faces: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """
    Draw access status overlay onto frame.

    Args:
        frame:  BGR image from cv2.
        status: One of 'ALLOW', 'WAIT', 'DENY'.
        name:   Recognised person's name (shown on ALLOW).
        score:  Similarity score (shown as confidence %).
        faces:  List of (x1, y1, x2, y2) face boxes to draw.

    Returns:
        Annotated frame (modified in-place and returned).
    """
    # ✅ FIX 1: Guard against None or empty frame
    if frame is None or frame.size == 0:
        raise ValueError("draw_status() received an empty or None frame.")

    # ✅ FIX 2: Normalise status — unknown values fall back to DENY safely
    status = status.upper().strip()
    if status not in _COLORS:
        status = "DENY"

    color = _COLORS[status]
    label = _LABELS[status]

    h, w = frame.shape[:2]

    # ── Tinted overlay ────────────────────────────────────────────────────────
    # ✅ FIX 3: Use dst=frame directly in addWeighted — original creates an
    #    extra full-frame copy every call with no benefit
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    cv2.addWeighted(overlay, _OVERLAY_ALPHA, frame, 1 - _OVERLAY_ALPHA, 0, frame)

    # ✅ FIX 4: Colored border around frame edge — more visible than tint alone
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, _BORDER_THICKNESS)

    # ── Status text ───────────────────────────────────────────────────────────
    # ✅ FIX 5: Centre text dynamically instead of hardcoded (50, 50)
    #    Hardcoded position clips on small frames and looks misaligned on large ones
    (text_w, text_h), baseline = cv2.getTextSize(
        label, _FONT, _FONT_SCALE, _FONT_THICKNESS
    )
    text_x = (w - text_w) // 2
    text_y = 50 + text_h  # top margin

    # ✅ FIX 6: Draw dark drop-shadow first for legibility on any background
    cv2.putText(frame, label, (text_x + 2, text_y + 2),
                _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICKNESS + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (text_x, text_y),
                _FONT, _FONT_SCALE, color, _FONT_THICKNESS, cv2.LINE_AA)

    # ── Name + confidence ─────────────────────────────────────────────────────
    # ✅ FIX 7: Show recognised name and score — silently discarded in original
    if name and status == "ALLOW":
        conf_text = f"{name}  ({score * 100:.1f}%)" if score is not None else name
        (cw, ch), _ = cv2.getTextSize(conf_text, _FONT, 0.7, 1)
        cx = (w - cw) // 2
        cy = text_y + ch + 16

        cv2.putText(frame, conf_text, (cx + 1, cy + 1),
                    _FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, conf_text, (cx, cy),
                    _FONT, 0.7, color, 1, cv2.LINE_AA)

    # ── Face boxes ────────────────────────────────────────────────────────────
    # ✅ FIX 8: Draw face bounding boxes passed in from detect_faces()
    #    Original module drew nothing around detected faces
    if faces:
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accent marks for a cleaner look
            corner_len = max(10, (x2 - x1) // 6)
            for px, py, dx, dy in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(frame, (px, py), (px + dx * corner_len, py), color, 3)
                cv2.line(frame, (px, py), (px, py + dy * corner_len), color, 3)

    # ✅ FIX 9: Return the frame — original returned None implicitly
    #    Callers doing `frame = draw_status(frame, ...)` got None back
    return frame