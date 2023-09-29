import cv2
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
import numpy as np
from tqdm import trange
from keras import models

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 128})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

"""
=========================================================================
LOADING THE DATA
=========================================================================
"""

"""Parameter"""
img_width, img_height = 28, 28
validation_data_dir = '../Images/test'
epochs = 20
batch_size = 32
num_class = 10

test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    class_mode='categorical')

"""
=========================================================================
Training and Saving the model and saving metrics in history
=========================================================================
"""

iteration = 0
for outer_iteration in range(0, 20):
    if outer_iteration == (iteration * 1):
        model = keras.models.load_model('../Data/MNIST_Iteration_' + str(outer_iteration) + '.h5')
        layer_outputs = [layer.output for layer in model.layers[:50]]
        activation_model = models.Model(inputs=model.input,
                                        outputs=layer_outputs)

        test_filenames = validation_generator.filepaths
        test_labels = validation_generator.labels

        Final_prediction_Test = np.zeros(len(test_filenames))
        print("\nLoop " + str(iteration) + " for the images:")
        previous_Class = 0
        previous_image = 0
        for class_Number in range(num_class):
            class0layer1 = []
            class0layer2 = []
            class0layer3 = []

            layer1 = np.zeros((512, 1))
            layer2 = np.zeros((512, 1))
            layer3 = np.zeros((num_class, 1))

            for image in trange(previous_image, (len(test_filenames)) - 1):
                name = test_filenames[image]
                test_IMAGE = np.asarray(cv2.resize(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2GRAY), (img_width, img_height))).flatten()
                test_IMAGE = np.asarray(test_IMAGE, dtype='int')
                test_IMAGE = test_IMAGE.reshape(1, (img_width * img_height))
                test_IMAGE = test_IMAGE.astype('float32')
                test_IMAGE /= 255
                Final_prediction_Test[image] = (int(model.predict_classes(test_IMAGE))==test_labels[image])

                img_tensor = test_IMAGE
                img_tensor = img_tensor.reshape(1, 784)
                activations = activation_model.predict(img_tensor)  # Returns the activation of the layers

                activation_1 = activations[1]
                activation_1 = np.reshape(activation_1, (512, 1))

                activation_2 = activations[4]
                activation_2 = np.reshape(activation_2, (512, 1))

                activation_3 = activations[7]
                activation_3 = np.reshape(activation_3, (num_class, 1))

                class0layer1.append(activation_1)
                class0layer2.append(activation_2)
                class0layer3.append(activation_3)

                if test_labels[image + 1] > previous_Class:
                    previous_Class += 1
                    previous_image = image + 1
                    break
            """
            =========================================================================
            Storing the activation values
            =========================================================================
            """
            name = "Test_Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_Number) + "_Layer1.csv"
            arr = np.reshape(np.asanyarray(class0layer1), (len(class0layer1), 512)).transpose()
            np.savetxt(name, arr, delimiter=",", fmt='%f', comments="")

            name = "Test_Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_Number) + "_Layer2.csv"
            arr = np.reshape(np.asanyarray(class0layer2), (len(class0layer2), 512)).transpose()
            np.savetxt(name, arr, delimiter=",", fmt='%f', comments="")

            name = "Test_Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_Number) + "_Layer3.csv"
            arr = np.reshape(np.asanyarray(class0layer3), (len(class0layer3), num_class)).transpose()
            np.savetxt(name, arr, delimiter=",", fmt='%f', comments="")

        iteration += 1

print("Simulation Complete!")
