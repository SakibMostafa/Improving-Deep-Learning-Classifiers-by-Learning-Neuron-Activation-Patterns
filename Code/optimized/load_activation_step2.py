# --- load_activations.py ---
# Loads activation data from CSVs and prepares training input

import numpy as np
import os

# Parameters
num_classes = 10
activation_dir = '..'  # Change if your CSVs are in a different path
layer_name = 'Layer3'
iteration = 0

X_train = []
y_train = []

print("Loading activation files...")

for class_number in range(num_classes):
    filename = f"Activation_Iteration_{iteration}_class{class_number}_{layer_name}.csv"
    filepath = os.path.join(activation_dir, filename)
    if os.path.exists(filepath):
        data = np.loadtxt(filepath, delimiter=',')
        # Each column is a sample
        data = data.T
        X_train.append(data)
        y_train.extend([class_number] * data.shape[0])
    else:
        print(f"Missing file: {filepath}")

X_train = np.vstack(X_train)
y_train = np.array(y_train)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Save for next stage or return as needed
np.save('../X_train.npy', X_train)
np.save('../y_train.npy', y_train)


