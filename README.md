# OpenCV Face Detection and Recognition Projects

This repository contains four different computer vision projects built using **Python** and **OpenCV**.  
Each project demonstrates a different technique for face detection and recognition.

---

# 1) Face Detection from Image (Haar Cascade)

## Description
This project detects human faces in a static image using the Haar Cascade classifier provided by OpenCV.

## Features
- Loads an image from disk  
- Converts the image to grayscale  
- Detects faces using Haar Cascade  
- Draws bounding boxes around detected faces  

## Technologies
- Python  
- OpenCV  
- Haar Cascade Classifier  

## How to Run
1. Place an image inside the project folder  
2. Update the image path in the code  
3. Run the script  

---

# 2) Face Recognition from Image (face_recognition Library)

## Description
This project recognizes known people in a group photo by comparing face encodings.

## Features
- Loads known person images  
- Extracts face embeddings  
- Detects faces in a group photo  
- Compares faces and assigns names  
- Displays results with bounding boxes  

## Technologies
- Python  
- OpenCV  
- face_recognition  

## How to Run
1. Add reference images for known persons  
2. Add a group photo  
3. Run the script  

---

# 3) Automatic Face Registration and Recognition (DeepFace)

## Description
This project automatically registers a user by collecting face data from different head directions and then performs face recognition using embeddings.

## Features
- Automatic face registration  
- Head pose detection (left, right, up, down, center)  
- Face embedding generation using FaceNet  
- Face recognition using cosine similarity  

## Technologies
- Python  
- OpenCV  
- DeepFace  
- NumPy  

## How to Run
1. Run the registration mode  
2. Follow on-screen instructions  
3. Run recognition mode  

---

# 4) Real-Time Mask Detection using Haar Cascade

## Description
This project detects whether a person is wearing a mask by checking if the mouth is visible inside the detected face region.

## Features
- Real-time webcam detection  
- Face detection using Haar Cascade  
- Mouth detection inside face region  
- Displays "Mask Detected" or "No Mask"  
- Bounding boxes and live status text  

## Technologies
- Python  
- OpenCV  
- Haar Cascade (Face + Mouth)  

## How to Run
1. Connect a webcam  
2. Run the script  
3. Press **Q** to exit  

---

## Installation

```bash
pip install opencv-python face-recognition deepface numpy
