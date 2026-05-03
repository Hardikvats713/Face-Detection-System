import cv2
import numpy as np
import config
from supabase import create_client
from recognition.detector import detect_faces
from recognition.embedder import get_embedding

try:
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Supabase client: {e}")

def enroll_student():
    name = input("Enter the student's name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    print("Opening camera... Position your face in the frame.")
    print("Press 'c' to CAPTURE and save your face.")
    print("Press 'q' or ESC to QUIT.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        try:
            faces = detect_faces(frame)
        except Exception as e:
            faces = []

        display_frame = frame.copy()
        
        # Draw boxes around detected faces
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'C' to Capture", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Enroll New Student", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if not faces:
                print("No face detected! Please try again.")
                continue
            
            print("Processing face...")
            # Use the first face detected
            x1, y1, x2, y2 = faces[0]
            face_crop = frame[y1:y2, x1:x2]
            
            try:
                emb = get_embedding(face_crop)
                # Convert numpy array to standard Python list so it can be sent to Supabase as JSON
                emb_list = emb.tolist()
                
                # Insert into Supabase
                data = {"name": name, "embedding": emb_list}
                supabase.table("students").insert(data).execute()
                print(f"\n✅ Successfully enrolled '{name}' and saved to Supabase!")
                break
            except Exception as e:
                print(f"\n❌ Error during enrollment or saving to Supabase: {e}")
                break

        elif key == ord('q') or key == 27:
            print("Enrollment cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    enroll_student()
