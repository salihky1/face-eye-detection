import cv2

# -------------------------------
# Initialize Camera
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# -------------------------------
# Load Haar Cascade Models
# -------------------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("mcs_mouth.xml")

if face_cascade.empty() or mouth_cascade.empty():
    print("Error: Cascade files could not be loaded.")
    exit()

print("Camera started. Press 'Q' to quit.")

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame could not be captured.")
        break

    # Flip image horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # Detect Faces
    # -------------------------------
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        cv2.putText(
            frame,
            "No face detected",
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
    else:
        for (x, y, w, h) in faces:

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Region of Interest (face area)
            face_gray = gray_frame[y:y + h, x:x + w]

            # -------------------------------
            # Detect Mouth inside Face ROI
            # -------------------------------
            mouths = mouth_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.4,
                minNeighbors=12,
                minSize=(30, 30)
            )

            if len(mouths) == 0:
                cv2.putText(
                    frame,
                    "Mask Detected",
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                for (mx, my, mw, mh) in mouths:
                    cv2.rectangle(
                        frame,
                        (x + mx, y + my),
                        (x + mx + mw, y + my + mh),
                        (150, 150, 150),
                        2
                    )

                cv2.putText(
                    frame,
                    "No Mask",
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    2
                )

    # -------------------------------
    # Display Frame
    # -------------------------------
    cv2.imshow("Mask Detection System", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------------
# Release Resources
# -------------------------------
cap.release()
cv2.destroyAllWindows()
