import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from keras.applications.resnet import preprocess_input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from tensorflow import keras
from keras.layers import Dense
import time
from keras import Input
from keras import backend as K
from tqdm import tqdm
import os
import shutil
BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Model / data parameters
img_width, img_height = 112, 112
input_shape = (img_width, img_height, 3)
train_data_dir = '../Images/train'
validation_data_dir = '../Images/test'
output_directory = '../Feature_Image'

epochs = 40
num_images_to_save = 50
batch_size = 64
num_class = 10
neuron_layer1 = 2048
training_ACC = np.zeros((epochs))
validation_ACC = np.zeros((epochs))

# Get model
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
# Specify Model
RN_model = ResNet50(include_top=False, weights=None, pooling='avg', input_tensor=Input(shape = input_shape))
X = RN_model.output
fc = Dense(num_class, activation='softmax', name='Output')(X)
model = Model(inputs=RN_model.inputs, outputs = fc)
model = load_model('Inception_50images_epoch5.h5')
# Data Generators
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    class_mode='categorical',
    batch_size=batch_size,
    color_mode='rgb')

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    class_mode='categorical',
    batch_size=batch_size,
    color_mode='rgb')

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
def loss_fn(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

@tf.function
def create_feature_vector(inputs, activation_model):
    feature_vector = tf.zeros((num_class, neuron_layer1), dtype=tf.float32)
    feature_step = 0
    for x_train, y_train in feature_generator:
        feature_step += 1
        if feature_step > int(feature_generator.n/batch_size):
            break
        x_train = np.expand_dims(x_train, axis=0)
        for image in range(0, len(y_train)):
            class_Num = np.argmax(y_train[image])
            indices_1 = []
            for val in range(0, neuron_layer1):
                indices_1.append([class_Num, val])
            train_IMAGE = x_train[:, image, :, :, :]
            feature_vector = tf.tensor_scatter_nd_add(feature_vector, indices_1, calculate_activation(train_IMAGE, activation_model))
    feature_vector = tf.math.divide(feature_vector, tf.constant(num_images_to_save, dtype=tf.float32))

    """----------------------Predict Values----------------------"""
    indices_1 = []
    for val in range(0, neuron_layer1):
        indices_1.append([val])
    batch, he, we, ch = inputs.shape
    calculated_predictions = tf.zeros((batch, num_class))
    inputs = tf.expand_dims(inputs, axis=0)
    for image in range(0, batch):
        feature_vector_sample = tf.zeros((neuron_layer1), dtype=tf.float32)
        img_tensor = inputs[:, image, :, :, :]
        feature_vector_sample = tf.tensor_scatter_nd_add(feature_vector_sample, indices_1, calculate_activation(img_tensor, activation_model))
        # Calculating the predictions
        indices_3 = []
        #we are making mistake
        for val in range(num_class):
            indices_3.append([image, val])
        calculated_predictions = tf.tensor_scatter_nd_update(calculated_predictions, indices_3, tf.reshape(tf.experimental.numpy.dot(feature_vector_sample, tf.transpose(feature_vector)), [-1]))
    return calculated_predictions, feature_vector

@tf.function
def test_prediction(inputs, feature_vector, activation_model):
    indices_1 = []
    for val in range(0, neuron_layer1):
        indices_1.append([val])
    batch, he, we, ch = inputs.shape
    calculated_predictions_test = tf.zeros((batch, num_class))
    inputs = tf.expand_dims(inputs, axis=0)
    for image in range(0, batch):
        feature_vector_sample = tf.zeros((neuron_layer1), dtype=tf.float32)
        img_tensor = inputs[:, image, :, :, :]
        feature_vector_sample = tf.tensor_scatter_nd_update(feature_vector_sample, indices_1, calculate_activation(img_tensor, activation_model))
        # Calculating the predictions
        indices_3 = []
        for val in range(num_class):
            indices_3.append([image, val])
        calculated_predictions_test = tf.tensor_scatter_nd_update(calculated_predictions_test, indices_3, tf.reshape(tf.experimental.numpy.dot(feature_vector, feature_vector_sample), [-1]))
        return calculated_predictions_test

@tf.function
def calculate_activation(train_IMAGE, activation_model):
    successive_feature_maps = activation_model((train_IMAGE), training=False)
    layer_names = [layer.name for layer in model.layers]
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if ("activation_93" in layer_name):
            display_grid = tf.reshape(feature_map[0, :], [-1])
            display_grid = tf.cast(display_grid > 0, tf.float32)
    return display_grid

@tf.function
def train_step(x, y, calculated_predictions):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        new_predictions = tf.math.divide(tf.math.multiply(calculated_predictions, predictions),
                                         tf.reduce_sum(tf.math.multiply(calculated_predictions, predictions),
                                                       axis=1, keepdims=True))
        loss_value = loss_fn(y, new_predictions)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(tf.argmax(y, axis=1), predictions)
    return loss_value

@tf.function
def test_step(x, y, calculated_predictions):
    predictions = model(x, training=False)
    new_predictions = tf.math.divide(tf.math.multiply(calculated_predictions, predictions),
                                     tf.reduce_sum(tf.math.multiply(calculated_predictions, predictions),
                                                   axis=1, keepdims=True))
    val_acc_metric.update_state(tf.argmax(y, axis=1), new_predictions)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
def feature_gen():
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    class_probabilities = {}
    predictions = model.predict(train_generator)
    for i in range(len(train_generator.filenames)):
        image_filename = train_generator.filenames[i]
        image_label = train_generator.classes[i]
        image_class = np.argmax(predictions[i,:])
        if image_label == image_class:
            image_probability = predictions[i, image_label]
        else:
            image_probability = 0

        if image_label not in class_probabilities:
            class_probabilities[image_label] = []
        class_probabilities[image_label].append((image_filename, image_probability))

    for class_label, image_probabilities in class_probabilities.items():
        class_folder = os.path.join(output_directory, str(class_label))
        os.makedirs(class_folder, exist_ok=True)

        # Sort the image_probabilities based on the prediction probability in descending order
        image_probabilities.sort(key=lambda x: x[1], reverse=True)

        # Save the top num_images_to_save images with the highest probabilities
        for j in range(min(num_images_to_save, len(image_probabilities))):
            image_filename, _ = image_probabilities[j]
            image_path = os.path.join(train_data_dir, image_filename)
            image_save_path = os.path.join(class_folder, os.path.basename(image_filename))
            shutil.copy2(image_path, image_save_path)
    feature_datagen = ImageDataGenerator(preprocessing_function = preprocess_input, validation_split=0.9)
    feature_generator = feature_datagen.flow_from_directory(
        output_directory,
        target_size=(img_width, img_height),
        class_mode='categorical',
        batch_size=batch_size,
        color_mode='rgb',
        shuffle = False
    )

    return feature_generator

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    step = 0
    start_time = time.time()
    feature_generator = feature_gen()
    model.save('Starting_Model_80images.h5', save_format='.h5')
    num_train_samples = train_generator.n
    # Iterate over the batches of the dataset.
    for x_batch_train, y_batch_train in train_generator:
        new_time = time.time()
        step += 1
        successive_outputs = [layer.output for layer in model.layers[1:]]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs, trainable=False)
        calculated_predictions, feature_vector_test = create_feature_vector(x_batch_train, activation_model)
        loss_value = train_step(x_batch_train, y_batch_train, calculated_predictions)
        # Log every 20 batches.
        if step % 10 == 0:
            #print(
            #    "Training loss (for one batch) at step %d: %.4f"
            #    % (step, float(tf.reduce_sum(loss_value)))
            #)
            print("Seen so far: %d samples" % ((step + 1) * batch_size), "for Epoch: ", epoch)
        if step > (num_train_samples//batch_size):
            break
        train_acc = train_acc_metric.result()
        train_acc = float(tf.reduce_sum(train_acc))
        #print("Training acc over step: %.4f" % (train_acc,))
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    train_acc = float(tf.reduce_sum(train_acc))
    training_ACC[epoch] = train_acc
    print(BLUE + "Training acc over epoch: %.4f" % (train_acc,) + RESET)

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    num_validation_samples = validation_generator.n
    validation_step = 0
    for x_batch_val, y_batch_val in validation_generator:
        validation_step += 1
        calculated_predictions, feature_vector_test = create_feature_vector(x_batch_val, activation_model)
        test_step(x_batch_val, y_batch_val, calculated_predictions)
        if validation_step > (num_validation_samples//batch_size):
            break

    val_acc = val_acc_metric.result()
    val_acc = float(tf.reduce_sum(val_acc))
    validation_ACC[epoch] = val_acc
    val_acc_metric.reset_states()
    print(GREEN + "Validation acc: %.4f" % (val_acc,) + RESET)
    print("Time taken for 1 Iteration: %.2fs" % (time.time() - start_time))
    model_name = 'Inception_50images_epoch' + str(epoch) + '.h5'
    model.save(model_name, save_format='h5')

print('-------------------------------------')
print('Training Accuracy: ', training_ACC)
print('-------------------------------------')
print('Validation Accuracy: ', validation_ACC)
