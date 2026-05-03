import cv2
import time
import logging
import sys

from database.supabase_client import load_students
from recognition.detector import detect_faces
from recognition.embedder import get_embedding
from recognition.matcher import match_face
from liveness.mediapipe_liveness import check_liveness, reset_liveness
from utils.beep import beep_async
from ui.display import draw_status

import config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Load known students ───────────────────────────────────────────────────────
# ✅ FIX 1: Wrap DB load in try/except — a Supabase failure should not silently
#    launch the gate with an empty known_embeddings list (everyone gets DENY)
try:
    known_embeddings, names = load_students()
    log.info(f"Loaded {len(names)} student(s) from database.")
except Exception as e:
    log.critical(f"Failed to load students: {e}")
    sys.exit(1)

# ── Camera setup ──────────────────────────────────────────────────────────────
# ✅ FIX 2: Validate camera opened successfully before entering loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    log.critical("Could not open camera (index 0). Check connection.")
    sys.exit(1)

# ✅ FIX 3: Set resolution via CAP_PROP instead of resizing every frame —
#    resizing after capture wastes CPU; setting props lets the driver do it
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── State ─────────────────────────────────────────────────────────────────────
last_seen:    dict[str, float] = {}   # name → timestamp first confirmed this session
marked:       set[str]         = set() # names already logged for attendance
unknown_start: float | None    = None  # when an unknown face first appeared
last_beep:    float            = 0.0

# ✅ FIX 4: Track active name across frames to detect subject change
#    and reset liveness state when a new person steps in front of camera
current_subject: str | None = None

log.info("Gate system running. Press ESC to exit.")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()

    # ✅ FIX 5: Distinguish camera error from end-of-stream
    if not ret or frame is None:
        log.warning("Failed to read frame — retrying.")
        time.sleep(0.05)
        continue

    now   = time.time()
    faces = detect_faces(frame)

    # ✅ FIX 6: Default status to DENY when no face is detected, not WAIT
    #    WAIT should mean "face found, still verifying" — no face = denied
    status       = "DENY" if not faces else "WAIT"
    display_name  = None
    display_score = None

    for (x1, y1, x2, y2) in faces:
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # ✅ FIX 7: Wrap per-face pipeline in try/except so one bad frame
        #    doesn't crash the entire gate
        try:
            emb          = get_embedding(face_crop)
            name, score  = match_face(emb, known_embeddings, names)
            live         = check_liveness(frame)
        except Exception as e:
            log.warning(f"Face processing error: {e}")
            continue

        # ✅ FIX 4 applied: Reset liveness when subject changes between frames
        if name != current_subject:
            log.info(f"Subject changed: {current_subject!r} → {name!r}")
            reset_liveness()
            current_subject = name
            last_seen.pop(name, None)  # restart confirm timer for new subject

        if name and live:
            status        = "ALLOW"
            display_name  = name
            display_score = score
            unknown_start = None

            # ✅ FIX 8: Start confirm timer on first sighting, not on every frame
            if name not in last_seen:
                last_seen[name] = now

            elapsed = now - last_seen[name]

            # ✅ FIX 9: Log attendance exactly once per session per person
            if elapsed >= config.CONFIRM_TIME and name not in marked:
                marked.add(name)
                log.info(f"ATTENDANCE MARKED: {name}  (score={score:.3f})")

        elif name and not live:
            # Known face but liveness not yet confirmed
            status        = "WAIT"
            display_name  = name
            display_score = score

        else:
            # Unknown face
            status = "DENY"

            if unknown_start is None:
                unknown_start = now

            time_unknown = now - unknown_start
            time_since_beep = now - last_beep

            if (time_unknown  > config.BEEP_DELAY and
                time_since_beep > config.BEEP_COOLDOWN):
                beep_async()
                last_beep = now
                log.warning(f"Unknown face at gate for {time_unknown:.1f}s — beep triggered.")

        # ✅ FIX 10: Only process the first (largest/most confident) face
        #    Multiple face boxes cause status to be overwritten by the last
        #    face in the list, making final status non-deterministic
        break

    # ✅ FIX 11: Receive returned frame — draw_status() returns annotated frame;
    #    original code discarded it (draw_status fix #9) so imshow showed raw frame
    frame = draw_status(frame, status,
                        name=display_name,
                        score=display_score,
                        faces=faces)

    cv2.imshow("AI Gate System", frame)

    key = cv2.waitKey(1) & 0xFF   # ✅ FIX 12: Mask to 8-bit — on Linux waitKey()
    if key == 27:                  #    can return >255 without masking, ESC never fires
        log.info("ESC pressed — shutting down.")
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
# ✅ FIX 13: Always release camera and destroy windows in a finally-style block
cap.release()
cv2.destroyAllWindows()
log.info(f"Session ended. Attendance marked for: {sorted(marked) or 'nobody'}")