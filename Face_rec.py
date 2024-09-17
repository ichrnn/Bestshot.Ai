import cv2
import dlib
import numpy as np


print("Script started")  # Add this at the start of your script

# Load face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download the model from dlib repo

# Function to detect blurriness using the variance of Laplacian
def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 100  # Threshold for blurriness; adjust as needed

# Function to check if eyes are open based on landmarks
def are_eyes_open(landmarks):
    left_eye_ratio = (landmarks[37][1] - landmarks[41][1]) / (landmarks[36][0] - landmarks[39][0])
    right_eye_ratio = (landmarks[43][1] - landmarks[47][1]) / (landmarks[42][0] - landmarks[45][0])
    return left_eye_ratio > 0.2 and right_eye_ratio > 0.2  # Adjust thresholds based on testing

# Function to detect faces, check orientation and eye state
def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in landmarks.parts()]

        # Check if eyes are open
        if not are_eyes_open(landmarks):
            print("Eyes are closed.")
            return False

        # Check if face is blurry
        if is_blurry(image):
            print("Face is blurry.")
            return False

        # Basic face orientation check (use more sophisticated pose estimation for production)
        nose = landmarks[30]
        chin = landmarks[8]
        if abs(nose[0] - chin[0]) > 10:  # Adjust tolerance for looking straight
            print("Face is not facing the camera.")
            return False

    return True

import os
import cv2

# Path to the folder containing images
folder_path = "/Volumes/Files/Face-REC/Pictures"

# List to store image file paths
image_files = []

# Iterate through the folder and gather image files (e.g., .jpg, .png)
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_files.append(os.path.join(folder_path, filename))

# Check if any image files were found
if not image_files:
    print("No images found in the directory.")
else:
    best_image = None
    for image_file in image_files:
        print(f"Processing {image_file}")
        image = cv2.imread(image_file)

        if image is None:
            print(f"Error: Could not load image {image_file}")
            continue

        # Call the process_image function here (assuming you have this function defined)
        if process_image(image):  # Assuming this is a function you have already defined
            best_image = image
            print(f"Best image selected: {image_file}")
            break

# If no suitable image is found
if best_image is None:
    print("No suitable image found.")

# If no suitable image found
if best_image is None:
    print("No suitable image found.")

# Show the best image if found
if best_image is not None:
    cv2.imshow("Best Image", best_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
print("Script finished")  # Add this at the end