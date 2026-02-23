import cv2

# -------------------------------
# Initialize Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# -------------------------------
# Load Haar Cascade Model
# -------------------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("Error: Face cascade file could not be loaded.")
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
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(60, 60)
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # -------------------------------
    # Display Result
    # -------------------------------
    cv2.imshow("Real-Time Face Detection", frame)

    # Press Q to exit
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

# -------------------------------
# Release Resources
# -------------------------------
cap.release()
cv2.destroyAllWindows()
