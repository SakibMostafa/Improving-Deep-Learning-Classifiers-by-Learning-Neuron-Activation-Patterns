# --- train_model_step0.py ---
# Trains a lightweight CNN model on CIFAR-10 and saves it

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Parameters
img_width, img_height, img_channels = 32, 32, 3
num_class = 10

# Load and normalize CIFAR-10 data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = y_train.flatten()
y_test = y_test.flatten()

# Define a lightweight CNN model
model = models.Sequential([
    layers.Input(shape=(img_width, img_height, img_channels)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_class, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Save the trained model
model.save("cifar10_light_model.h5")
print("Model saved as 'cifar10_light_model.h5'")