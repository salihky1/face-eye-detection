import cv2
import numpy as np
from deepface import DeepFace
import os
import time

SAVE_DIR = "user_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------
# Helper Functions
# -------------------------------
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_embeddings():
    known_faces = {}
    for file in os.listdir(SAVE_DIR):
        if file.endswith(".npy"):
            name = file.replace("_embeddings.npy", "")
            known_faces[name] = np.load(os.path.join(SAVE_DIR, file), allow_pickle=True)
    return known_faces

def get_head_direction(angles):
    yaw, pitch, roll = angles  # left-right, up-down, tilt
    if yaw > 15:
        return "Turn Right"
    elif yaw < -15:
        return "Turn Left"
    elif pitch > 12:
        return "Look Down"
    elif pitch < -12:
        return "Look Up"
    else:
        return "Look Center"

# -------------------------------
# Camera Opening Function (Safe for Mac)
# -------------------------------
def open_camera():
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"🎥 Camera {i} selected.")
            return cap
    print("⚠️ No camera found.")
    return None

# -------------------------------
# 1️⃣ Automatic Face Registration
# -------------------------------
def register_face(name):
    embeddings = []
    directions_done = set()
    required_directions = {"Turn Right", "Turn Left", "Look Up", "Look Down", "Look Center"}

    cap = open_camera()
    if cap is None:
        return

    cv2.namedWindow("Registration", cv2.WINDOW_NORMAL)
    print(f"\nStarting face registration for {name}...")
    print("The system will automatically detect head directions (ESC to cancel).")

    last_save_time = 0

    while len(directions_done) < len(required_directions):
        ret, frame = cap.read()
        if not ret:
            print("Camera frame could not be read.")
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['head_pose'], enforce_detection=True)
            angles = (
                analysis[0]["head_pose"]["yaw"],
                analysis[0]["head_pose"]["pitch"],
                analysis[0]["head_pose"]["roll"]
            )
            direction = get_head_direction(angles)

            cv2.putText(frame, f"Direction: {direction}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            cv2.imshow("Registration", frame)

            if direction not in directions_done and time.time() - last_save_time > 2.5:
                emb = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)
                embeddings.append(emb[0]["embedding"])
                directions_done.add(direction)
                print(f"{direction} completed ✅")
                last_save_time = time.time()

        except Exception:
            cv2.putText(frame, "Face not detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Registration", frame)

        if cv2.waitKey(10) & 0xFF == 27:  # ESC
            print("Registration canceled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    file_path = os.path.join(SAVE_DIR, f"{name}_embeddings.npy")
    np.save(file_path, embeddings)
    print(f"\n🎉 Automatic registration completed for {name}! Data saved to {file_path}")

# -------------------------------
# 2️⃣ Face Recognition
# -------------------------------
def recognize_face():
    known_faces = load_embeddings()
    if not known_faces:
        print("⚠️ No registered faces found. Please register first.")
        return

    cap = open_camera()
    if cap is None:
        return

    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    print("🎥 Camera started... (Press ESC to exit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame could not be read.")
            break

        try:
            emb = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)[0]["embedding"]

            best_name = "Unknown"
            best_score = -1

            for name, data in known_faces.items():
                similarities = [cosine_similarity(emb, e) for e in data]
                score = np.mean(similarities)
                if score > best_score:
                    best_score = score
                    best_name = name

            threshold = 0.55
            text = best_name if best_score > threshold else "Unknown"

            cv2.putText(frame, f"{text} ({best_score:.2f})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if text != "Unknown" else (0, 0, 255), 3)
        except Exception:
            cv2.putText(frame, "Face not detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Main Menu
# -------------------------------
def main():
    print("""
=============================
      FACE SYSTEM v2.1 (macOS)
=============================
1️⃣  Automatic Face Registration
2️⃣  Face Recognition
0️⃣  Exit
""")

    while True:
        choice = input("Your choice: ").strip()
        if choice == "1":
            name = input("Enter the name to register: ").strip()
            register_face(name)
        elif choice == "2":
            recognize_face()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
