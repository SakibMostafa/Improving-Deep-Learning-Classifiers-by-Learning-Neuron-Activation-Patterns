# --- train_custom_model_step3.py ---
# Trains a model using activation inputs and custom loss logic

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tqdm import trange

# Load activation features
X_train = np.load('../X_train.npy').astype(np.float32)
y_train = np.load('../y_train.npy')

num_classes = 10
input_dim = X_train.shape[1]
epochs = 20
batch_size = 64

# Convert labels to one-hot
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)

# Build a simple classification model
model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(num_classes, activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

# Compute class activation vectors
def compute_class_vectors(X, y):
    vectors = np.zeros((num_classes, input_dim), dtype=np.float32)
    counts = np.zeros(num_classes)
    for i in range(len(X)):
        vectors[y[i]] += X[i]
        counts[y[i]] += 1
    for c in range(num_classes):
        if counts[c] > 0:
            vectors[c] /= counts[c]
    return tf.convert_to_tensor(vectors, dtype=tf.float32)

feature_vector = compute_class_vectors(X_train, y_train)

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[idx]
    y_train_shuffled = y_train_onehot[idx]

    for i in range(0, len(X_train), batch_size):
        x_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)

            # Custom projection using class feature vectors
            dot_sim = tf.matmul(x_batch, tf.transpose(feature_vector))  # shape: (batch, num_classes)
            weights = tf.nn.softmax(dot_sim)
            adjusted_preds = preds * weights
            adjusted_preds /= tf.reduce_sum(adjusted_preds, axis=1, keepdims=True)

            loss = loss_fn(y_batch, adjusted_preds)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_acc_metric.update_state(y_batch, adjusted_preds)

    acc = train_acc_metric.result()
    print(f"Training Accuracy: {acc:.4f}")
    train_acc_metric.reset_states()

print("Training complete.")
model.save("custom_activation_classifier.h5")
