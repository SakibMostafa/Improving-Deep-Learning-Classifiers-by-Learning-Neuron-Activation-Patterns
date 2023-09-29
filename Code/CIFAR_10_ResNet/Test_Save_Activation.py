import cv2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.resnet import preprocess_input
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import trange

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 128})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Model / data parameters
img_width, img_height = 112, 112
input_shape = (img_width, img_height, 1)
train_data_dir = '../Images/train'
validation_data_dir = '../Images/test'
epochs = 20
batch_size = 32
num_class = 10

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

"""
=========================================================================
Training and Saving the model and saving metrics in history
=========================================================================
"""

iteration = 14
for outer_iteration in range(0, 20):
    if outer_iteration == (iteration * 1):
        model = keras.models.load_model('../Data/CIFAR_ResNet_Iteration_' + str(outer_iteration) + '.h5')
        layer_outputs = [layer.output for layer in model.layers[:350]]
        successive_outputs = [layer.output for layer in model.layers[1:]]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

        test_filenames = validation_generator.filepaths
        test_labels = validation_generator.labels

        """
        =========================================================================
        Calculating the activations
        =========================================================================
        """
        print("\nLoop " + str(iteration) + " for the images:")
        previous_Class = 0
        previous_image = 0
        for class_Number in range(num_class):
            """
            =========================================================================
            Empty Array and Matrix for storing activation values
            =========================================================================
            """
            class0layer1 = []
            class0layer2 = []
            class0layer7 = []

            layer1 = np.zeros((512, 1))
            layer2 = np.zeros((2048, 1))
            layer7 = np.zeros((10, 1))

            for image in trange(previous_image, (len(test_filenames)) - 1):
                name = test_filenames[image]
                test_IMAGE = preprocess_input(np.asarray(
                    cv2.resize(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB),
                               (img_width, img_height)), dtype='float32'))
                test_IMAGE = np.expand_dims(test_IMAGE, axis=0)
                img_tensor = test_IMAGE
                successive_feature_maps = activation_model.predict(img_tensor)  # Returns the activation of the layers

                layer_names = [layer.name for layer in model.layers]
                layer_count = 0
                ii = 0
                for layer_name, feature_map in zip(layer_names, successive_feature_maps):
                    if "relu" in layer_name:
                        ii += 1
                        if ii >= 32:
                            layer_count += 1
                            n_features = feature_map.shape[-1]
                            size = feature_map.shape[1]
                            display_grid = np.zeros((size, size * n_features))
                            for i in range(n_features):
                                x = feature_map[0, :, :, i]
                                display_grid[:, i * size: (i + 1) * size] = x
                            if layer_count == 1:
                                display_grid = np.reshape(display_grid.flatten(), (512, 1))
                                activation_1 = display_grid
                            elif layer_count == 2:
                                display_grid = np.reshape(display_grid.flatten(), (2048, 1))
                                activation_2 = display_grid
                display_grid = np.reshape(np.array(successive_feature_maps[175]).flatten(), (10, 1))
                activation_7 = display_grid
                class0layer1.append(activation_1)
                class0layer2.append(activation_2)
                class0layer7.append(activation_7)

                if test_labels[image + 1] > previous_Class:
                    previous_Class += 1
                    previous_image = image + 1
                    break
            """
            =========================================================================
            Storing the activation values
            =========================================================================
            """
            name = "../Data_CNN/Test_Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_Number) + "_Layer1.csv"
            arr = np.reshape(np.asanyarray(class0layer1), (len(class0layer1), 512)).transpose()
            np.savetxt(name, arr, delimiter=",", fmt='%f', comments="")

            name = "../Data_CNN/Test_Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_Number) + "_Layer2.csv"
            arr = np.reshape(np.asanyarray(class0layer2), (len(class0layer2), 2048)).transpose()
            np.savetxt(name, arr, delimiter=",", fmt='%f', comments="")

            name = "../Data_CNN/Test_Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_Number) + "_Layer3.csv"
            arr = np.reshape(np.asanyarray(class0layer7), (len(class0layer7), 10)).transpose()
            np.savetxt(name, arr, delimiter=",", fmt='%f', comments="")

        iteration += 1

print("Simulation Complete!")
