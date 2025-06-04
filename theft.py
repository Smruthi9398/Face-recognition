import cv2
import numpy as np
import os
from datetime import datetime

# Load the face recognizer and face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Directory where the face data is stored
data_dir = 'face rec/face_data'

def train_model():
    images = []
    labels = []
    label_dict = {}
    target_size = (200, 200)  # Resize all images to this size

    for i, file in enumerate(os.listdir(data_dir)):
        image_path = os.path.join(data_dir, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, target_size)  # Resize the image
        label = file.split('_')[0]
        
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        
        images.append(image)
        labels.append(label_dict[label])

    images = np.array(images)
    labels = np.array(labels)
    recognizer.train(images, labels)
    return label_dict

def recognize_face():
    cap = cv2.VideoCapture(0)
    label_dict = train_model()
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face)

            for name, id in label_dict.items():
                if id == label:
                    person_name = name
                    break

            if confidence < 50:
                record_face(person_name)
                cv2.putText(frame, f'{person_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) == 27:  # Escape key
            break

    cap.release()
    cv2.destroyAllWindows()

def record_face(name):
    with open('face.txt', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{name},{dt_string}\n')
        print(f'face recorded for {name} at {dt_string}')

recognize_face()