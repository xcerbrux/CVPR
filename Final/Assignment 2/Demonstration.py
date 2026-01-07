import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # helps avoid NaNs in some builds

import cv2 as cv
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import tensorflow as tf
import time

print("Loading model...")
model = load_model('F:/CVPR/Final/face_recognition_model.keras', compile=False)
print("Model loaded successfully!")

class_names = [
    '21-45902-3', '22-46138-1', '22-46139-1', '22-46258-1', '22-46342-1',
    '22-46536-1', '22-46590-1', '22-46887-1', '22-46983-1', '22-47542-2',
    '22-47813-2', '22-47884-2', '22-47892-2', '22-47898-2', '22-48005-2',
    '22-48021-2', '22-48023-2', '22-48091-2', '22-48133-2', '22-48205-2',
    '22-48569-3', '22-48582-3', '22-48666-3', '22-48682-3', '22-48725-3',
    '22-48833-3', '22-48841-3', '22-48915-3', '22-49037-3', '22-49068-3',
    '22-49196-3', '22-49331-3', '22-49338-3', '22-49355-3', '22-49507-3',
    '22-49538-3', '22-49575-3', '22-49643-3', '22-49783-3', '22-49791-3',
    '22-49824-3', '22-49843-3', '22-49862-3', '23-50254-1', '23-50277-1',
    '23-50346-1', '23-51127-1', '23-51308-1'
]

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if face_cascade.empty():
    print("Error: Could not load face cascade classifier")
    exit()

padding_ratio = 0.3
prediction_queue = deque(maxlen=10)
unknown_count = 0

webcam = cv.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam")
    exit()

webcam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

print("\nStarting face recognition...")
print("Press 'q' to quit")

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, frame.shape[1])
        y2 = min(y + h + pad_y, frame.shape[0])

        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue

        face_img_resized = cv.resize(face_img, (256, 256))
        face_img_rgb = cv.cvtColor(face_img_resized, cv.COLOR_BGR2RGB)

        face_img_array = face_img_rgb.astype("float32") / 255.0
        face_img_array = np.expand_dims(face_img_array, axis=0)
        face_img_array = np.clip(face_img_array, 0.0, 1.0)

        logits = model.predict(face_img_array, verbose=0)

        # DEBUG — tells us if the model is healthy
        print("DEBUG:", logits.shape, "min:", float(np.min(logits)), "max:", float(np.max(logits)))

        # Apply softmax if outputs aren't probabilities
        if np.any(logits < 0) or np.any(logits > 1):
            probs = tf.nn.softmax(logits, axis=-1).numpy()
        else:
            probs = logits

        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))

        prediction_queue.append(predicted_class)

        if prediction_queue:
            most_common_prediction = max(
                set(prediction_queue),
                key=prediction_queue.count
            )
        else:
            most_common_prediction = predicted_class

        if confidence > 0.6:
            unknown_count = 0
            class_name = class_names[most_common_prediction]
            color = (0, 255, 0)
        else:
            unknown_count += 1
            class_name = "Unknown" if unknown_count > 5 else ""
            color = (0, 0, 255)

        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if class_name:
            cv.putText(
                frame,
                f'{class_name} ({confidence:.2f})',
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv.imshow('Webcam Face Recognition', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print("\nQuitting...")
        break

webcam.release()
cv.destroyAllWindows()
print("Program ended.")
