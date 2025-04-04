﻿# Facial-Emotional-Detection-Using-Neural-Networks
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
   ![Screenshot (334)](https://github.com/user-attachments/assets/e7c19d34-a84b-4c08-beab-0c0e7978695f)
   ## 3.2 Sad
   ![Screenshot (336)](https://github.com/user-attachments/assets/7220826f-d77f-463e-815c-82718cef6706)
   ## 3.3 Surprise
   ![image](https://github.com/user-attachments/assets/26170ecb-fbaa-41e8-928f-af3c384d6262)
 

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

### Contributions

Contributions, issues, and feature requests are welcome. Feel free to fork the repository and create a pull request.

---

This version excludes any setup instructions while retaining the project's core details for a concise README. Let me know if you need further edits!
