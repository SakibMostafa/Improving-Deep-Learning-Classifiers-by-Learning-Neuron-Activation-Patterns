# --- extract_activations.py ---
# Loads the trained model and extracts intermediate layer activations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
from tqdm import trange

# Load the saved model
model = keras.models.load_model("cifar10_light_model.h5")

# CIFAR-10 Parameters
img_width, img_height, img_channels = 32, 32, 3
num_class = 10

# Load CIFAR-10 again
(_, y_train), (_, _) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
(x_train, _), _ = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.

# Create activation model
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'dense' in layer.name]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

iteration = 0
Final_prediction = np.zeros(len(x_train))

print("\nLoop", iteration, "for the CIFAR-10 images:")

for class_Number in range(num_class):
    class_activations = [[] for _ in range(len(layer_outputs))]

    class_indices = np.where(y_train == class_Number)[0]

    for idx in trange(min(100, len(class_indices))):
        i = class_indices[idx]
        image = x_train[i:i+1]
        label = y_train[i]

        pred_class = model.predict(image, verbose=0).argmax()
        Final_prediction[i] = (pred_class == label)

        activations = activation_model.predict(image, verbose=0)

        for layer_idx, activation in enumerate(activations):
            flattened = activation.flatten().reshape(-1, 1)
            class_activations[layer_idx].append(flattened)

    for layer_idx, layer_act in enumerate(class_activations):
        arr = np.reshape(np.array(layer_act), (len(layer_act), -1)).T
        fname = f"Activation_Iteration_{iteration}_class{class_Number}_Layer{layer_idx+1}.csv"
        np.savetxt(fname, arr, delimiter=",", fmt='%f')

np.savetxt(f"Prediction_Results_Iteration_{iteration}_.csv", Final_prediction, delimiter=",", fmt="%f")
print("Simulation Complete!")
