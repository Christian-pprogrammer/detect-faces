import cv2
import os

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image
img = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Loop through the faces and crop and save each one
for i, (x, y, w, h) in enumerate(faces):
    # Crop the face from the image
    face = img[y:y+h, x:x+w]
    
    # Save the face to a file
    filename = f'face_{i}.jpg'
    cv2.imwrite(filename, face)

# Print a message indicating how many faces were detected and saved
num_faces = len(faces)
print(f'{num_faces} faces detected and saved to disk.')
