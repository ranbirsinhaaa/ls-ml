import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
data = image_dataset_from_directory(
    'path_to_your_dataset',  # Replace with the path to your dataset directory
    image_size=(256, 256),
    batch_size=32,
    label_mode='int'
)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Visualize the labels given to different animals
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()
data = data.map(lambda x, y: (x / 255.0, y))
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
test_size = int(0.1 * len(data))

train_data = data.take(train_size)
val_data = data.skip(train_size).take(val_size)
test_data = data.skip(train_size + val_size).take(test_size)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes for elephants, tigers, cheetahs, and crocodiles
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # You can adjust this based on the dataset size and performance
)
loss, accuracy = model.evaluate(test_data)
print(f"Test accuracy: {accuracy}")

# Do not change this code
if accuracy >= 0.85:
    print(f"Congratulations, CNN assignment complete!! Your accuracy is {accuracy}")
else:
    print(f"Try again, not enough accuracy! Your accuracy is {accuracy}")
