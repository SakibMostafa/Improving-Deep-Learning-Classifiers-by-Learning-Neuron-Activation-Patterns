import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils import np_utils
import tensorflow as tf
import keras
from tqdm import trange

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU': 128} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

img_width, img_height = 28, 28
train_data_dir = '../Images/train'
validation_data_dir = '../Images/test'
epochs = 20
batch_size = 32
num_class = 10

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.02))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.02))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    class_mode='categorical')

test_datagen = ImageDataGenerator()
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    class_mode='categorical')

print('Loading Testing Data')
image_ID_Test = np.transpose(np.array([np.array(validation_generator.filepaths), validation_generator.labels]))
test_Filename = image_ID_Test[:,0]
y_test = np.asarray(image_ID_Test[:,1], dtype=float)
Y_test = np_utils.to_categorical(y_test, num_class)
X_test = [cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY), (img_width, img_height)) for file in test_Filename]
X_test = np.asarray(X_test, dtype='int')
X_test = X_test.reshape(len(y_test), (img_width * img_height))
X_test = X_test.astype('float32')

print('\nLoading Training Data')
image_ID_Train = np.transpose(np.array([np.array(train_generator.filepaths), train_generator.labels]))
train_Filename = image_ID_Train[:, 0]
y_train = np.asarray(image_ID_Train[:, 1], dtype=float)
Y_train = np_utils.to_categorical(y_train, num_class)
X_train = [cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY), (img_width,img_height)) for file in train_Filename]
X_train = np.asarray(X_train, dtype='int')
X_train = X_train.reshape(len(y_train), (img_width * img_height))
X_train = X_train.astype('float32')

accuracy_Values = np.asarray(np.zeros((epochs, 2)))
model.summary()
for ite in range(0,20):
    print('Current Iteration: ', ite)
    history1 = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=2,
                        shuffle=True,
                        validation_data=(X_test, Y_test))
    model_name = '../Data/MNIST_Iteration_' + str(ite) + '.h5'
    model.save(model_name)

    train_Acc = history1.history['accuracy']
    train_Acc = train_Acc[0]
    test_Acc = history1.history['val_accuracy']
    test_Acc = test_Acc[0]
    print('Training Accuracy: ', train_Acc, ' and Testing Accuracy: ', test_Acc)
    accuracy_Values[ite, 0] = float(train_Acc)
    accuracy_Values[ite, 1] = float(test_Acc)

file_P = "../Data/Training_Testing_Accuracy.csv"
np.savetxt(file_P, accuracy_Values, delimiter=",", fmt="%f", header="Training, Testing", comments="")