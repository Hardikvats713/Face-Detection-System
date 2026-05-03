import base64
import numpy as np
import cv2
import logging
import sys
import os
import time
import threading
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recognition.detector import detect_faces
from recognition.embedder import get_embedding
from recognition.matcher import match_face
import config
from liveness.mediapipe_liveness import check_liveness
import time

liveness_db = {}
from supabase import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    log.info("Supabase client initialized.")
except Exception as e:
    log.critical(f"Failed to initialize Supabase: {e}")
    sys.exit(1)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

class StudentCache:
    def __init__(self):
        self.embeddings = None
        self.names = []
        self.data = []
        self.last_update = 0
        self.ttl = 300  # Cache for 5 minutes
        self.lock = threading.Lock()

    def get_data(self):
        with self.lock:
            if time.time() - self.last_update > self.ttl or self.embeddings is None:
                log.info("Refreshing student cache from Supabase...")
                try:
                    res = supabase.table("students").select("*").execute()
                    if not res.data:
                        return None, [], []
                    
                    emb_list = []
                    name_list = []
                    data_list = []
                    
                    for row in res.data:
                        if row.get("embedding") is None or row.get("name") is None:
                            continue
                        try:
                            embedding = np.array(row["embedding"], dtype=np.float32)
                            emb_list.append(embedding)
                            name_list.append(row["name"])
                            data_list.append(row)
                        except (ValueError, TypeError):
                            continue
                    
                    if emb_list:
                        self.embeddings = np.stack(emb_list)
                        self.names = name_list
                        self.data = data_list
                        self.last_update = time.time()
                    else:
                        self.embeddings = None
                except Exception as e:
                    log.error(f"Error fetching from Supabase for cache: {e}")
            
            return self.embeddings, self.names, self.data

    def invalidate(self):
        with self.lock:
            self.last_update = 0

student_cache = StudentCache()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/register", methods=["POST"])
def register():
    try:
        name   = request.form.get("name", "").strip()
        erp    = request.form.get("erp", "").strip()
        course = request.form.get("course", "").strip()
        year   = request.form.get("year", "").strip()

        if not name:
            return jsonify({"success": False, "message": "Name is required."}), 400

        image_front = request.files.get("image_front")
        image_left = request.files.get("image_left")
        image_right = request.files.get("image_right")

        if not image_front:
            return jsonify({"success": False, "message": "Front image is required."}), 400

        images_to_process = []
        if image_front: images_to_process.append(("Front", image_front))
        if image_left: images_to_process.append(("Left Side", image_left))
        if image_right: images_to_process.append(("Right Side", image_right))

        success_count = 0
        
        for label, img_file in images_to_process:
            img_bytes = img_file.read()
            if not img_bytes:
                continue
                
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            faces = detect_faces(frame)
            if not faces:
                return jsonify({"success": False, "message": f"No face detected in the {label} photo. Please upload a clear photo."}), 400

            x1, y1, x2, y2 = faces[0]
            face_crop = frame[y1:y2, x1:x2]
            emb = get_embedding(face_crop)
            emb_list = emb.tolist()

            data = {"name": name, "erp": erp, "course": course, "year": year, "embedding": emb_list}
            supabase.table("students").insert(data).execute()
            success_count += 1
        
        # Invalidate cache so new students are loaded immediately
        student_cache.invalidate()
        
        log.info(f"Enrolled student: {name} with {success_count} face angles.")

        return jsonify({"success": True, "message": f"Successfully enrolled {name} using {success_count} angles!"})

    except Exception as e:
        log.error(f"Registration error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
def verify():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False, "status": "ERROR", "message": "No image data received."}), 400

        image_data = data["image"]
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
        else:
            encoded = image_data

        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "status": "ERROR", "message": "Could not decode the image."}), 400

        faces = detect_faces(frame)
        if not faces:
            log.info("Verify: NO FACE detected")
            return jsonify({"success": False, "status": "NO_FACE", "message": "No face detected. Position your face clearly."})

        log.info(f"Verify: found {len(faces)} face(s)")

        # FETCH FROM CACHE INSTEAD OF DB
        known_embeddings, names_list, students_data = student_cache.get_data()
        
        if known_embeddings is None or len(known_embeddings) == 0:
            return jsonify({"success": False, "status": "EMPTY_DB", "message": "No students in database."})

        # Cleanup old liveness sessions
        now = time.time()
        for k in list(liveness_db.keys()):
            if now - liveness_db[k]['last_seen'] > 10:
                del liveness_db[k]

        results = []
        for face in faces:
            x1, y1, x2, y2 = face
            face_crop = frame[y1:y2, x1:x2]
            
            # Skip faces that are too small or produce bad embeddings
            try:
                emb = get_embedding(face_crop)
            except ValueError as ve:
                log.warning(f"Skipping face at ({x1},{y1},{x2},{y2}): {ve}")
                results.append({
                    "status": "ACCESS_DENIED",
                    "message": "Face too small or blurry.",
                    "score": 0.0,
                    "box": [int(x1), int(y1), int(x2), int(y2)]
                })
                continue
            
            name, score = match_face(emb, known_embeddings, names_list)
            log.info(f">>> VERIFY: name={name}, score={score:.4f}, threshold={config.THRESHOLD}")

            if name:
                # ── Liveness / Anti-Spoofing Check ────────────────────
                # Pass person_name for temporal (multi-frame) tracking
                is_live, reason, liveness_details = check_liveness(
                    face_crop, full_frame=frame,
                    face_box=[int(x1), int(y1), int(x2), int(y2)],
                    person_name=name
                )
                
                # is_live: True=live, False=spoof, None=still verifying
                if is_live is None:
                    status = "VERIFYING"
                elif is_live:
                    status = "ACCESS_GRANTED"
                else:
                    status = "SPOOF_DETECTED"
                
                log.info(f">>> LIVENESS: status={status}, reason={reason}")

                if status == "VERIFYING" or status == "SPOOF_DETECTED":
                    results.append({
                        "status": status,
                        "message": reason,
                        "name": name,
                        "score": round(float(score), 3),
                        "box": [int(x1), int(y1), int(x2), int(y2)]
                    })
                elif status == "ACCESS_GRANTED":
                    matched = next((s for s in students_data if s["name"] == name), {})
                    results.append({
                        "status": "ACCESS_GRANTED",
                        "name": name,
                        "score": round(float(score), 3),
                        "erp": matched.get("erp", "N/A"),
                        "course": matched.get("course", "N/A"),
                        "year": matched.get("year", "N/A"),
                        "box": [int(x1), int(y1), int(x2), int(y2)]
                    })
            else:
                results.append({
                    "status": "ACCESS_DENIED",
                    "message": "Face not recognized.",
                    "score": round(float(score), 3),
                    "box": [int(x1), int(y1), int(x2), int(y2)]
                })

        return jsonify({
            "success": True,
            "status": "MULTIPLE_FACES",
            "results": results,
            "count": len(faces)
        })

    except Exception as e:
        log.error(f"Verification error: {e}", exc_info=True)
        return jsonify({"success": False, "status": "ERROR", "message": str(e)}), 500


@app.route("/api/students", methods=["GET"])
def list_students():
    try:
        res = supabase.table("students").select("id, name, erp, course, year").execute()
        
        # We might have duplicates if they registered 3 times. We can filter them visually.
        unique_students = []
        seen = set()
        if res.data:
            for s in res.data:
                if s["name"] not in seen:
                    seen.add(s["name"])
                    unique_students.append(s)
        
        return jsonify({"success": True, "students": unique_students})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    log.info("Starting AI Attendance System on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)




