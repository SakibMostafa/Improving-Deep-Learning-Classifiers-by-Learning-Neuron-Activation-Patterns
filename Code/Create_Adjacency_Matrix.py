import os
import numpy as np
import pandas as pd
from tqdm import trange

Layer_Num = 2           #Select the layer you want to calculate the adjacency matrix
top_Values = 50         #Top S neurons from each class
Total_Neurons = 512     #Total Neurons in the layer
number_Classes = 10     #Total Number of classes

for Layer_Num in range(1,4):
    for iteration in range(0, 20):
        final_Avg_Activation = np.zeros((Total_Neurons, 1))
        final_Activation_Freq = np.zeros((Total_Neurons, 1))
        print("\nLayer : ", Layer_Num, " Iteration: ", iteration)
        for class_num in trange(0, number_Classes):
            file_path = os.path.join("../Data/Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_num) + "_Layer" + str(Layer_Num) + ".csv")
            dataset = pd.read_csv(file_path, header=None)
            dr, dc = dataset.shape
            activation_val = np.array(
                dataset.iloc[0:dr, 0:dc])  # activation_val stores the activavtion of the neurons in a class in a layer

            temp_activation_val = np.copy(activation_val)
            temp_activation_val[temp_activation_val > 0] = 1
            temp_val = np.reshape(np.average(temp_activation_val, axis=1), (Total_Neurons, 1))
            final_Activation_Freq = np.append(final_Activation_Freq, temp_val, axis=1)

            r, c = activation_val.shape
            avg_Activation = []
            # Iterate over all the neurons
            for row in range(r):
                temp_Activation = []
                # For each neuron go over all the images store the activation value of only the activated neurons
                for column in range(c):
                    if activation_val[row, column] > 0:
                        temp_Activation = np.append(temp_Activation, activation_val[
                            row, column])  # temp_Activation stores the activation value of the activated neruons only

                if np.sum(temp_Activation) != 0:
                    avg_Activation = np.append(avg_Activation, np.average(temp_Activation))
                elif np.sum(temp_Activation) == 0:
                    avg_Activation = np.append(avg_Activation, 0)

            """Apply threshold to the average activation value"""
            activation_Threshold = 0.000005
            avg_Activation[avg_Activation < activation_Threshold] = 0
            avg_Activation = np.reshape(avg_Activation, (Total_Neurons, 1))
            final_Avg_Activation = np.append(final_Avg_Activation, avg_Activation, axis=1)

        Avg_Activation_Weight = final_Avg_Activation[:,
                                1:number_Classes+1]  # Avg_Activation_Weight stores the average activation value of the neurons for different classes
        Avg_Activation_Frequency = final_Activation_Freq[:,
                                   1:number_Classes+1]  ##Avg_Activation_Frequency stores the average activation frequency of the neurons for different classes

        """
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        So far I have stored the average activation frequency and average activation weight in Avg_Activation_Frequency, Avg_Activation_Weight respectively
        ----------------------------------------------------------------------------------------------------------------------------------------------------
        """
        network_Edge_weights = np.zeros((Total_Neurons, Total_Neurons))
        for class_num in range(0, number_Classes):

            """
            ---------------------------------------------------------
            Select frequency or weight to do the calculation
            ---------------------------------------------------------
            """
            temp_Weight = Avg_Activation_Frequency[:, class_num]
            top_N_values = top_Values + 2
            ind = np.argpartition(temp_Weight, -(top_N_values))[-(top_N_values):]
            ind = np.sort(ind)

            # Loading the activation values for a class to calculate the edge weight based on activation
            #file_path = os.path.join("Activation_Value_Iteration_" + str(iteration) + "_class" + str(class_num) + "_Layer" + str(Layer_Num) + ".csv")
            dataset = pd.read_csv(file_path, header=None)
            dr, dc = dataset.shape
            activation_val = np.array(dataset.iloc[0:dr, 0:dc])

            r, c = activation_val.shape
            index_Iteration = 0
            network_Edge_weights = np.zeros((Total_Neurons, Total_Neurons))
            # This loop goes over all the neurons
            for row in range(0, r):
                if index_Iteration < top_N_values:
                    # Find if the neuron is in top N neurons
                    if ind[index_Iteration] == row:
                        inner_index_iteration = np.copy(index_Iteration)
                        index_Iteration = index_Iteration + 1
                        # As the index is sorted, in this loop start from the next neuron and go till last neuron
                        for inner_row in range(row, r):
                            # Find the next Top N neuron
                            if (inner_index_iteration < top_N_values - 2) and (ind[inner_index_iteration + 1] == inner_row):
                                inner_index_iteration = inner_index_iteration + 1
                                # Iterate over all the images to find how many times both the neurons are activated
                                for column in range(0, c):
                                    # Find if both the consecutive neurons were activated
                                    if activation_val[row, column] > temp_Weight[row] and activation_val[
                                        inner_row, column] > temp_Weight[inner_row]:
                                        network_Edge_weights[row, inner_row] = network_Edge_weights[row, inner_row] + 1

            file_P = os.path.join("../Adjacency_Matrix/Layer_" +str(Layer_Num)+ "/" +
                                  "Top_" + str(top_Values) + "_Edge_Frequency_Iteration_" + str(iteration) + "_class_" + str(class_num) + ".csv")
            #Normalized the value before saving
            np.savetxt(file_P, network_Edge_weights / c, delimiter=",", fmt='%f', comments="")

print("\n\nSimulation Complete! ")
