import cv2
import face_recognition

# -------------------------------
# Load Known Face Images
# -------------------------------

person1_image = face_recognition.load_image_file("person1.jpg")
person1_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file("person2.jpg")
person2_encoding = face_recognition.face_encodings(person2_image)[0]

# Store known face encodings and names
known_face_encodings = [person1_encoding, person2_encoding]
known_face_names = ["Alice", "Bob"]


# -------------------------------
# Load Test Image (Group Photo)
# -------------------------------

test_image = face_recognition.load_image_file("group.jpg")
test_image_cv = cv2.imread("group.jpg")

# Find face locations and encodings in the test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)


# -------------------------------
# Process Each Detected Face
# -------------------------------

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    # Compare detected face with known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown Person"

    # If a match is found, use the first matched face
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw rectangle around face
    cv2.rectangle(
        test_image_cv,
        (left, top),
        (right, bottom),
        (255, 0, 0),
        2
    )

    # Write name above the rectangle
    cv2.putText(
        test_image_cv,
        name,
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    print(f"Detected: {name}")

# -------------------------------
# Show Result
# -------------------------------

cv2.imshow("Face Recognition Result", test_image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
