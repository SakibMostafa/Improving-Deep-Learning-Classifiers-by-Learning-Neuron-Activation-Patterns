import os
import numpy as np
import pandas as pd
from math import log2
from tqdm import trange

Layer_Num = 2
number_Divisions = 10           #Total R partitions
number_Classes = 10             #Total Number of classes
number_Iteration = 20
num_Layer = 1
entropies = np.zeros((number_Iteration, number_Classes))

for iteration in range(0,number_Iteration):
    entropy_Histogram = np.zeros((1,number_Divisions))
    print("Iteration: ", iteration)
    for class_num in range(0,number_Classes):
        file_path = os.path.join("../Data/Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_num) + "_Layer" + str(Layer_Num) + ".csv")
        dataset = pd.read_csv(file_path, header=None)
        dr, dc = dataset.shape
        activation_val_Layer_1 = np.array(dataset.iloc[0:dr, 0:dc])         #activation_val stores the activavtion of the neurons in a class in a layer
        activation_val = activation_val_Layer_1.transpose()

        if num_Layer > 2:
            for layers in range(3, num_Layer+1):
                file_path = os.path.join("../Data/Activation_Value_Iteration_" + str(iteration) + "_class" + str(
                    class_num) + "_Layer" + str(layers) + ".csv")
                dataset = pd.read_csv(file_path, header=None)
                dr, dc = dataset.shape
                activation_val_Layer_n = np.array(
                    dataset.iloc[0:dr,
                    0:dc]).transpose()  # activation_val stores the activavtion of the neurons in a class in a layer
                activation_val = np.concatenate((activation_val, activation_val_Layer_n), axis=1)
        """
        Normalizing Activations to sum to 1 and normal entropy
        """

        dr, dc = activation_val.shape
        print('Class Num: ', class_num)
        print("Samples: ", dr, "Columns: ", dc)
        for ite2 in range(dc):
            if np.sum(activation_val[:,ite2]) != 0:
                activation_val[:,ite2] = activation_val[:,ite2] / np.sum(activation_val[:,ite2])
        final_Val = np.copy(activation_val)
        num_Images, num_Neurons = final_Val.shape
        entropy_All = 0
        for ite in trange(num_Neurons):
            #ONE EXTRA PARTITION JUST TO KEEP THE ZERO VALUES.
            images_in_Divisions = np.zeros((number_Divisions+1))
            max_Value = np.max(final_Val[:, ite])
            partition_Values = max_Value / number_Divisions
            if np.sum(final_Val[:,ite]) != 0:
                for ite2 in range(num_Images):
                    start, end = 0, partition_Values
                    for ite3 in range(0, number_Divisions):
                        if final_Val[ite2, ite] == 0:
                            images_in_Divisions[0] += 1
                            break
                        elif (final_Val[ite2, ite] > start) and (final_Val[ite2, ite] < end):
                            images_in_Divisions[ite3+1] += 1
                            break
                        else:
                            start = np.copy(end)
                            end = (ite3+2) * partition_Values
                images_in_Divisions = images_in_Divisions[1:]
                images_in_Divisions /= np.sum(images_in_Divisions)
                images_in_Divisions [images_in_Divisions == 0] = 1
                images_in_Divisions[np.isnan(images_in_Divisions)] = 1
                temp_2 = np.zeros((1,number_Divisions))
                for val in range(len(images_in_Divisions)):
                    temp_2[0,val] = images_in_Divisions[val]
                entropy_Histogram = np.concatenate((entropy_Histogram, temp_2), axis=0)
                for ite4 in images_in_Divisions:
                    entropy_All -= (ite4 * log2(ite4))
            else:
                entropy_All += 0
        entropies[iteration, class_num] = entropy_All
        print("Entropy: ", entropy_All)
        file_Ps = os.path.join("../Results/Per_Class_Entropy_Values_Per_Iteration_"+str(iteration)+"_Class_" + str(class_num) + ".csv")
        np.savetxt(file_Ps, entropy_Histogram[1:], delimiter=",", fmt='%f', comments="")
file_P = os.path.join("../Results/Entropy/Per_Class/Entropy_Values_Per_Iteration_Per_Class.csv")
np.savetxt(file_P, entropies, delimiter=",", fmt='%f', header = "Class_0, Class_1, Class_2, Class_3, Class_4, Class_5, Class_6, Class_7, Class_8, Class_9", comments = "")

print("Simulation Complete! ")

