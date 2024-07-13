import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            img = img.resize((64, 64))
            images.append(np.array(img))
            label = 0 if 'class0' in filename else 1  # Assume filename has class info
            labels.append(label)
    return np.array(images), np.array(labels)
images, labels = load_images_from_folder('path_to_images_folder')
images = images / 255.0  # Normalize pixel values
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
