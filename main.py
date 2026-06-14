import cv2
import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name = 'buffalo_l')
app.prepare(ctx_id = 0, det_size = (640, 640))

known_faces_dir = r"C:\Users\User\Pictures\faces"

known_encodings = []
known_names = []

for personName in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, personName)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)
            faces = app.get(img)
            if faces:
                known_encodings.append(faces[0].embedding)
                known_names.append(personName)
                print(f"Encoded: {personName}/{filename}")

print(f"Loaded {len(known_encodings)} faces")

video = cv2.VideoCapture(0)

while True:
    success, frame = cv2.VideoCapture(0).read()
    if not success:
        break
