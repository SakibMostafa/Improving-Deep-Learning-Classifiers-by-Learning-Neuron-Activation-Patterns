import os

import keras
import numpy as np
import pandas as pd
from tqdm import trange
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import tensorflow as tf
accuracy_Values = np.asarray(np.zeros((20, 2)))

def normalize(arr):
    arr = arr/arr.sum()
    return arr

Layer_Num = 1
Layer_Num_2 = 2
num_Layer = 2
number_Classes = 10
num_Iteration = 20

training = np.zeros((num_Iteration))
testing = np.zeros((num_Iteration))
training_multi = np.zeros((num_Iteration))
testing_multi = np.zeros((num_Iteration))
training_multi_V2 = np.zeros((num_Iteration))
testing_multi_V2 = np.zeros((num_Iteration))

for iteration in range(0,num_Iteration):
    print('Iteration: ', iteration)
    a = np.ones((11)) * -1
    b = np.ones((10)) * -1
    a_test = np.ones((11)) * -1
    b_test = np.ones((10)) * -1
    real_class_label = [0]
    real_class_label_Test = [0]
    for class_num in range(0, number_Classes):
        file_path = os.path.join(
            "../Data/Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_num) + "_Layer" + str(
                Layer_Num) + ".csv")
        dataset = pd.read_csv(file_path, header=None)
        dr, dc = dataset.shape
        activation_val_Layer_1 = np.array(dataset.iloc[0:dr, 0:dc])
        activation_val_Layer_1 = activation_val_Layer_1.transpose()
        temp_class_Label = np.ones((dc)) * class_num

        file_path = os.path.join("../Data/Activation_Value_Iteration_" + str(iteration) + "_class" + str(
            class_num) + "_Layer" + str(Layer_Num_2) + ".csv")
        dataset = pd.read_csv(file_path, header=None)
        dr, dc = dataset.shape
        activation_val_Layer_2 = np.array(dataset.iloc[0:dr, 0:dc])
        activation_val_Layer_2 = activation_val_Layer_2.transpose()
        activation_val_Class = np.concatenate((activation_val_Layer_1, activation_val_Layer_2), axis=1)

        """LAST LAYER CALCULATION"""
        file_path_Last = os.path.join("../Data/Activation_Value_Iteration_" + str(iteration) + "_class" + str(
            class_num) + "_Layer" + str(num_Layer+1) + ".csv")
        dataset_Last = pd.read_csv(file_path_Last, header=None)
        dr, dc = dataset_Last.shape
        activation_val_Layer_Last = np.array(dataset_Last.iloc[0:dr, 0:dc])
        activation_val_Layer_Last = activation_val_Layer_Last.transpose()
        if class_num == 0:
            activation_val_Class_Last = activation_val_Layer_Last
        else:
            activation_val_Class_Last = np.concatenate((activation_val_Class_Last, activation_val_Layer_Last), axis=0)

        if num_Layer > 2:
            for layers in range(3, num_Layer+1):
                file_path = os.path.join("../Data/Activation_Value_Iteration_" + str(
                    iteration) + "_class" + str(class_num) + "_Layer" + str(layers) + ".csv")
                dataset = pd.read_csv(file_path, header=None)
                dr, dc = dataset.shape
                activation_val_Layer_n = np.array(dataset.iloc[0:dr, 0:dc])
                activation_val_Layer_n = activation_val_Layer_n.transpose()
                activation_val_Class = np.concatenate((activation_val_Class, activation_val_Layer_n), axis=1)
        if class_num == 0:
            shape0, shape1 = activation_val_Class.shape
            activation_val = np.zeros((1,shape1))

        """
        ''''''''''''''''''''''''''''Testing Data'''''''''''''''''''''''''''''
        """
        file_path_Test = os.path.join(
            "../Data/Test_Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_num) + "_Layer" + str(
                Layer_Num) + ".csv")
        dataset_Test = pd.read_csv(file_path_Test, header=None)
        dr, dc = dataset_Test.shape
        activation_val_Layer_1_Test = np.array(dataset_Test.iloc[0:dr, 0:dc])
        activation_val_Layer_1_Test = activation_val_Layer_1_Test.transpose()
        temp_class_Label_Test = np.ones((dc)) * class_num

        file_path_Test = os.path.join("../Data/Test_Activation_Value_Iteration_" + str(
            iteration) + "_class" + str(class_num) + "_Layer" + str(Layer_Num_2) + ".csv")
        dataset_Test = pd.read_csv(file_path_Test, header=None)
        dr, dc = dataset_Test.shape
        activation_val_Layer_2_Test = np.array(dataset_Test.iloc[0:dr, 0:dc])
        activation_val_Layer_2_Test = activation_val_Layer_2_Test.transpose()
        activation_val_Class_Test = np.concatenate((activation_val_Layer_1_Test, activation_val_Layer_2_Test), axis=1)

        if num_Layer > 2:
            for layers in range(3, num_Layer + 1):
                file_path_Test = os.path.join("../Data/Test_Activation_Value_Iteration_" + str(iteration) + "_class" + str(
                    class_num) + "_Layer" + str(layers) + ".csv")
                dataset_Test = pd.read_csv(file_path_Test, header=None)
                dr, dc = dataset_Test.shape
                activation_val_Layer_n_Test = np.array(dataset_Test.iloc[0:dr, 0:dc])
                activation_val_Layer_n_Test = activation_val_Layer_n_Test.transpose()
                activation_val_Class_Test = np.concatenate((activation_val_Class_Test, activation_val_Layer_n_Test), axis=1)
        if class_num == 0:
            shape0, shape1 = activation_val_Class_Test.shape
            activation_val_Test = np.zeros((1, shape1))

        """LAST LAYER TEST CALCULATION"""
        file_path_Last_Test = os.path.join("../Data/Test_Activation_Value_Iteration_" + str(iteration) + "_class" + str(
            class_num) + "_Layer" + str(num_Layer + 1) + ".csv")
        dataset_Last_Test = pd.read_csv(file_path_Last_Test, header=None)
        dr, dc = dataset_Last_Test.shape
        activation_val_Layer_Last_Test = np.array(dataset_Last_Test.iloc[0:dr, 0:dc])
        activation_val_Layer_Last_Test = activation_val_Layer_Last_Test.transpose()
        if class_num == 0:
            activation_val_Class_Last_Test = activation_val_Layer_Last_Test
        else:
            activation_val_Class_Last_Test = np.concatenate((activation_val_Class_Last_Test, activation_val_Layer_Last_Test), axis=0)

        """
        Probability Calculation
        """
        activation_val = np.concatenate((activation_val, activation_val_Class), axis=0)
        activation_val_Test = np.concatenate((activation_val_Test, activation_val_Class_Test), axis=0)

        sam, neu = activation_val_Class.shape
        activation_val_Class[activation_val_Class > 0] = 1
        temp_column_Sum = np.sum(activation_val_Class, axis=0) / sam
        if class_num == 0:
            column_Sum = np.copy(temp_column_Sum)
        else:
            column_Sum = np.vstack((column_Sum, temp_column_Sum))

        real_class_label = np.append(real_class_label, temp_class_Label)
        real_class_label_Test = np.append(real_class_label_Test, temp_class_Label_Test)

    activation_val = activation_val[1:,:]
    activation_val_Test = activation_val_Test[1:,:]
    num_Samples, num_Neurons = activation_val.shape
    num_Samples_Test, num_Neurons_Test = activation_val_Test.shape
    predicted_class_label = np.zeros((num_Samples, number_Classes))
    predicted_class_label_Test = np.zeros((num_Samples_Test, number_Classes))

    predicted_class_label_Last = np.zeros((num_Samples, number_Classes))
    predicted_class_label_Last_V2 = np.zeros((num_Samples, number_Classes))
    predicted_class_label_Test_Last = np.zeros((num_Samples_Test, number_Classes))
    predicted_class_label_Test_Last_V2 = np.zeros((num_Samples_Test, number_Classes))

    real_class_label = real_class_label[1:]
    real_class_label_Test = real_class_label_Test[1:]
    train_STDV = np.std(activation_val_Class_Last, axis=1)
    for sample in trange(num_Samples):
        current_Sample = np.copy(activation_val[sample, :])
        current_Sample_Last = np.copy(activation_val_Class_Last[sample, :])
        current_Sample[current_Sample > 0] = 1
        predicted_class_label[sample, :] = normalize(np.dot(column_Sum, current_Sample))
        predicted_class_label_Last[sample, :] = normalize((np.dot(column_Sum, current_Sample) * current_Sample_Last))

        if sample < num_Samples_Test:
            current_Sample_Test = np.copy(activation_val_Test[sample, :])
            current_Sample_Test_Last = np.copy(activation_val_Class_Last_Test[sample, :])
            current_Sample_Test[current_Sample_Test > 0] = 1
            predicted_class_label_Test[sample, :] = normalize(np.dot(column_Sum, current_Sample_Test))
            predicted_class_label_Test_Last[sample, :] = normalize(np.dot(column_Sum, current_Sample_Test) * current_Sample_Test_Last)

    x_train = np.append((predicted_class_label_Last), (activation_val_Class_Last), axis=1)
    x_test = np.append((predicted_class_label_Test_Last), (activation_val_Class_Last_Test), axis=1)

    Y_train = np_utils.to_categorical(real_class_label, number_Classes)
    Y_test = np_utils.to_categorical(real_class_label_Test, number_Classes)

    model = Sequential()
    model.add(Dense(number_Classes+number_Classes, input_shape=((number_Classes + number_Classes),)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(number_Classes))
    model.add(Activation('relu'))
    model.add(Dense(number_Classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    if iteration == 0:
        model.summary()
    for ite in range(0, 5):
        print('Current Iteration: ', ite)
        history1 = model.fit(x_train, Y_train,
                             batch_size = 8,
                             epochs=1,
                             verbose=2,
                             shuffle=True,
                             validation_data=(x_test, Y_test))

        train_Acc = history1.history['accuracy']
        train_Acc = train_Acc[0]
        test_Acc = history1.history['val_accuracy']
        test_Acc = test_Acc[0]
        print('Training Accuracy: ', train_Acc, ' and Testing Accuracy: ', test_Acc)
    accuracy_Values[iteration, 0] = float(train_Acc)
    accuracy_Values[iteration, 1] = float(test_Acc)
    tf.keras.backend.clear_session()
print(accuracy_Values)

print('  ', accuracy_Values)
file_P = "../Data/Training_Testing_Custom_CNN_Accuracy.csv"
np.savetxt(file_P, accuracy_Values, delimiter=",", fmt="%f", header="Training, Testing", comments="")