import cv2

# -------------------------------
# Initialize Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# -------------------------------
# Load Haar Cascade Models
# -------------------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
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

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # Detect Faces
    # -------------------------------
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Region of Interest (Face Area)
        face_color = frame[y:y + h, x:x + w]
        face_gray = gray_frame[y:y + h, x:x + w]

        # -------------------------------
        # Detect Eyes inside Face ROI
        # -------------------------------
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

    # -------------------------------
    # Display Result
    # -------------------------------
    cv2.imshow("Face and Eye Detection", frame)

    # Press Q to exit
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# -------------------------------
# Release Resources
# -------------------------------
cap.release()
cv2.destroyAllWindows()
