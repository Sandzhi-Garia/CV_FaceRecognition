import cv2
import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from scipy.datasets import face

app = FaceAnalysis(name = 'buffalo_l')
app.prepare(ctx_id = 0, det_size = (640, 640))

known_faces_dir = r"C:\Users\User\Pictures\faces"

known_encodings = []# List to store known face encodings
known_names = []#List of names corresponding to the known encodings

for personName in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, personName)
    if not os.path.isdir(person_dir):#True if the path is a directory, False if it's a file
        continue
    for filename in os.listdir(person_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(person_dir, filename)#full path to the image file
            img = cv2.imread(img_path)
            faces = app.get(img)#Detect faces in the image and get their embeddings
            if faces:#Check if any faces were detected
                known_encodings.append(faces[0].embedding)#first found face
                known_names.append(personName)
                print(f"Encoded: {personName}/{filename}")

print(f"Loaded {len(known_encodings)} faces")

video = cv2.VideoCapture(0)

while True:
    success, frame = cv2.VideoCapture(0).read()
    if not success:
        break
    faces = app.get(frame)
    for face in faces:#Computing for each detected face
        embedding = face.embedding
        name = "Unknown"
        if known_encodings:#Check if there are known encodings to compare with
            similarities = [np.dot(embedding, known_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(known_embedding)) 
                            for known_embedding in known_encodings]#cosine similarity between embeddings
            best_match = np.argmax(similarities)
            if similarities[best_match] > 0.4:#minimum similarity threshold for recognition
                name = known_names[best_match]
        box = face.bbox.astype(int)#Bounding box coordinates for the detected face
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()