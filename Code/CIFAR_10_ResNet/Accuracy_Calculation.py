import numpy as np
import pandas as pd
from numpy import genfromtxt

number_Classes = 10
num_Iteration = 20
accuracy_Values = np.asarray(np.zeros((20, 2)))

for iteration in range(0,num_Iteration):
    file_path = "../Data_/Prediction_Results_Iteration_" + str(iteration) + "_.csv"
    dataset = genfromtxt(file_path, delimiter=",")
    accuracy_Values[iteration, 0] = np.average(dataset)
    file_path = "../Data_/Prediction_Results_Iteration_Test_" + str(iteration) + "_.csv"
    dataset = genfromtxt(file_path, delimiter=",")
    accuracy_Values[iteration, 1] = np.average(dataset)

print('  ', accuracy_Values)
file_P = "../Data_/Training_Testing_Accuracy.csv"
np.savetxt(file_P, accuracy_Values, delimiter=",", fmt="%f", header="Training, Testing", comments="")