import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('santhosh.keras')  # Update with the actual path to your saved model

# Define custom emotion labels and create a dictionary to map them to numerical indices
emotion_mapping = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Surprised': 5, 'Sad': 6}

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change if you have multiple cameras

# Initialize face detection using the haarcascade_frontalface_default.xml file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    while True:
        ret, frame = cap.read()  # Read frame from webcam

        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if any faces are detected
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

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

            # Display emotion prediction and percentages on the face region
            emotion_percentage = np.max(predictions) * 100
            text = f"{emotion}: {emotion_percentage:.2f}%"
            cv2.putText(frame, text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame with face detection, emotion prediction, and percentages
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
