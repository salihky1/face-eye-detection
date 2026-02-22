import cv2

# -------------------------------
# Load Image from File
# -------------------------------
image_path = "hogileyuztanima/images.jpeg"
image = cv2.imread(image_path)

# Check if image is loaded correctly
if image is None:
    print("Error: Image could not be loaded.")
    exit()

# -------------------------------
# Load Haar Cascade Classifier
# -------------------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Check if cascade file is loaded
if face_cascade.empty():
    print("Error: Haar cascade file could not be loaded.")
    exit()

# -------------------------------
# Convert Image to Grayscale
# -------------------------------
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -------------------------------
# Detect Faces
# -------------------------------
faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.3,
    minNeighbors=4,
    minSize=(30, 30)
)

print(f"Number of faces detected: {len(faces)}")

# -------------------------------
# Draw Rectangles Around Faces
# -------------------------------
for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    # Optional: write text above face
    cv2.putText(
        image,
        "Face",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# -------------------------------
# Show the Result
# -------------------------------
cv2.imshow("Face Detection Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
