import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
import tkinter as tk

# Load the trained model
model = load_model('santhosh.keras')

# Define custom emotion labels and create a dictionary to map them to numerical indices
emotion_mapping = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Surprised': 5, 'Sad': 6}

def detect_emotions(name):
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam, change if you have multiple cameras
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize face detection using the haarcascade_frontalface_default.xml file
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    message_displayed = False
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            break

        if not message_displayed:
            frame = display_message(frame, f"Face detection for {name}")
            message_displayed = True

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_roi_gray = np.expand_dims(face_roi_gray, axis=-1)
            face_roi_gray = np.expand_dims(face_roi_gray, axis=0)
            face_roi_gray = face_roi_gray.astype('float32') / 255.0

            predictions = model.predict(face_roi_gray)
            predicted_class = np.argmax(predictions)
            emotion = [key for key, value in emotion_mapping.items() if value == predicted_class][0]

            emotion_percentage = np.max(predictions) * 100
            text = f"{emotion}: {emotion_percentage:.2f}%"
            cv2.putText(frame, text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if name:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"Emotion detected for {name} at {timestamp}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def display_message(frame, message):
    message_frame = np.zeros((70, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(message_frame, message, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    combined_frame = np.vstack((message_frame, frame))
    return combined_frame

def get_name():
    name = name_entry.get()
    print(f"Entered Name: {name}")
    window.destroy()
    detect_emotions(name)

# Create a GUI window
window = tk.Tk()
window.title("Face Detector name")

# Create a label for the message with styling
message_label = tk.Label(window, text="Enter your name:", font=("Arial", 14), padx=20, pady=10)
message_label.pack()

# Create a text box for entering the name with styling
name_entry = tk.Entry(window, font=("Arial", 14), bd=3, relief=tk.SOLID)
name_entry.pack(padx=20, pady=10)

# Create a button to submit the entered name with styling
submit_button = tk.Button(window, text="Submit", font=("Arial", 14), command=get_name, padx=10, pady=5)
submit_button.pack(padx=20, pady=10)

window.mainloop()
