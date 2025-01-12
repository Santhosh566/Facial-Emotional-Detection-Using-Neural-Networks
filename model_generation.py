import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array

# Load your CSV dataset into a pandas DataFrame
df = pd.read_csv("fer.csv")

# Extract emotion labels, pixel values, and usages from the DataFrame
emotions = df['emotion']
pixels = df['pixels']
usages = df['Usage']
num_classes = 7
# Convert pixels to image arrays and normalize them
def process_pixels(pixel_string):
    pixel_array = np.array(pixel_string.split(), dtype='float32')
    pixel_array = pixel_array.reshape((48, 48, 1))  # Assuming images are grayscale
    pixel_array /= 255.0  # Normalize pixel values
    return pixel_array

X = np.array([process_pixels(pixel) for pixel in pixels])
y = np.array(emotions)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)

# Save the trained model to a file
model.save('face.keras')
#run che e code first