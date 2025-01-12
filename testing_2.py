import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('santhosh.h5')  # Update with the actual path to your saved model

# Define custom emotion labels and create a dictionary to map them to numerical indices
emotion_mapping = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Surprised': 5, 'Sad': 6}

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change if you have multiple cameras

try:
    while True:
        ret, frame = cap.read()  # Read frame from webcam

        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = frame[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))  # Resize face region to match model input size
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            face_roi_gray = np.expand_dims(face_roi_gray, axis=-1)  # Add channel dimension
            face_roi_gray = np.expand_dims(face_roi_gray, axis=0)  # Add batch dimension
            face_roi_gray = face_roi_gray.astype('float32') / 255.0  # Normalize pixel values

            # Predict emotion using the model
            predictions = model.predict(face_roi_gray)
            predicted_class = np.argmax(predictions)
            emotion = [key for key, value in emotion_mapping.items() if value == predicted_class][0]

            # Display emotion prediction on the face region
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame with face detection and emotion prediction
        cv2.imshow('Emotion Detection', frame)

        # Press 'q' to exit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
