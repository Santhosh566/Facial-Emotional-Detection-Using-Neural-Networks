# Facial-Emotional-Detection-Using-Neural-Networks
# Facial Emotion Detection Using Neural Networks

This project uses Convolutional Neural Networks (CNNs) to detect facial emotions from images. It leverages a preprocessed dataset of facial expressions and provides functionalities for emotion prediction using a trained model, integrated into a Flask-based web application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Web Application Features](#web-application-features)
- [Screenshots](#screenshots)
- [License](#license)

---

## Project Overview

The Facial Emotion Detection project aims to classify facial expressions into one of seven emotions: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Surprised**, and **Sad**. The trained model can be used in real-time webcam applications or via a web-based interface.

---

## Features

- **Deep Learning**: Utilizes a CNN for accurate emotion detection.
- **Web Integration**: Flask-based web application for emotion prediction and user interactions.
- **Database Integration**: PostgreSQL database for user authentication and feedback storage.
- **Real-Time Detection**: Integration with OpenCV for live webcam predictions.
- **Scalability**: Designed with modularity for easy extension.

---

## Technologies Used

- **Python Libraries**: TensorFlow, Keras, NumPy, Pandas, Scikit-learn
- **Flask Framework**: Web development and API integration
- **Database**: PostgreSQL
- **Frontend**: HTML, CSS, Bootstrap (for templates)
- **Computer Vision**: OpenCV

---

## Dataset

The dataset used is **FER (Facial Emotion Recognition)**. It contains labeled facial images in grayscale (48x48 resolution).
Data Set link:https://www.kaggle.com/datasets/msambare/fer2013
**Classes**:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprised

---

## Model Architecture

The CNN model is composed of:
1. Convolutional layers for feature extraction
2. MaxPooling layers for dimensionality reduction
3. Dense layers for classification

The final layer uses a **softmax** activation function to predict emotion probabilities.
## CNN Layers
![image](https://github.com/user-attachments/assets/5dcaeaa7-79f1-47cd-a325-fa441af6a0bf)
## Model Accuracy
![image](https://github.com/user-attachments/assets/da22d448-3f80-40c5-a4bd-dcd34097516a)
---
## TRAINING AND VALIDATION ACCURACY
![image](https://github.com/user-attachments/assets/5fefc6a6-64d0-4d01-83da-b28d4c8f6e86)
## TRAINING AND VALIDATION LOSS
![image](https://github.com/user-attachments/assets/d69efa5d-b550-4c7d-bf2a-5e882798f590)
## Web Application Features

- **Login/Registration System**: User authentication with Flask-WTF.
- **Emotion Prediction**: Upload images for emotion analysis.
- **Contact Form**: Submit messages or feedback via the web app.

---

## Screenshots

1. **Home Page**  
 ![image](https://github.com/user-attachments/assets/f0328f7f-3ff3-42ce-a8fe-030b04cb4102)
2. **Real-Time Detection page**
   ![image](https://github.com/user-attachments/assets/12696f53-3aee-4ccf-92c0-82fa073d2df2)
3. Real-Time Detections
   ## 3.1 Happy
   ![Happy](https://github.com/user-attachments/assets/770ca031-7bd9-4b0b-8518-7303c6588565)
   ## 3.2 Sad
   ![sad](https://github.com/user-attachments/assets/29f2f3dd-1d48-4741-94bf-12cd4c0a5daa)
   ## 3.3 Surprise
   ![image](https://github.com/user-attachments/assets/26170ecb-fbaa-41e8-928f-af3c384d6262)
   ## 3.4 Angry
   ![angry](https://github.com/user-attachments/assets/6448c6ed-6b6b-4167-9f65-6336b831b85f)
   ## 3.5 Fear
   ![fear](https://github.com/user-attachments/assets/a9f448ce-a8c2-4ee2-b7b6-cb3d2cac8e3f)
   ## 3.6 Disgust
   ![disgust](https://github.com/user-attachments/assets/60225e3d-7e78-4b8a-987e-f68f7172a825)
   ## 3.7 Neutral
   ![neutral](https://github.com/user-attachments/assets/e11c3c04-ab1c-4ff6-b826-25048d0fe6c3)

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

### Contributions

Contributions, issues, and feature requests are welcome. Feel free to fork the repository and create a pull request.

---

This version excludes any setup instructions while retaining the project's core details for a concise README. Let me know if you need further edits!
