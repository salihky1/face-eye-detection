import cv2

# -------------------------------
# Load Image from Disk
# -------------------------------
image_path = "images1111.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image could not be loaded.")
    exit()

# -------------------------------
# Load Haar Cascade Models
# -------------------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    print("Error: Cascade files could not be loaded.")
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
    minSize=(60, 60)
)

print(f"Number of faces detected: {len(faces)}")

# -------------------------------
# Process Each Face
# -------------------------------
for (x, y, w, h) in faces:

    # Draw rectangle around face
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    # Region of Interest (Face Area)
    face_color = image[y:y + h, x:x + w]
    face_gray = gray_image[y:y + h, x:x + w]

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
            (0, 0, 255),
            2
        )

# -------------------------------
# Show Result
# -------------------------------
cv2.imshow("Face and Eye Detection (Image)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
