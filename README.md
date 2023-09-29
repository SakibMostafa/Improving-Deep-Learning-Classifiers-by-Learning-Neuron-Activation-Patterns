# Leveraging Neuron Activation Patterns to Explain and Improve Deep Learning Classifiers

This code explains ways of analyzing the neuron activation pattern proposed in PID_4708.


## Requirements
Python 3.7 or later\

see `requirement.txt` for exact authors environment.

Before starting to train the model save the training data in `Images/train` and the testing data in `Images/test`.


## Calculating Entropy

After storing the activation values using `python Train_Save_Activation.py`, we can use the following code to calculate the entropy of all the entire dataset for different epochs using the code as follows

```
python Entropy_All_Class.py
```

It will save the entropy values of each iteration in `Results/Entropy/All_Class`.

If we want to calculate the entropy of individual classes we can use the code as follows

```
python Entropy_Per_Class.py
```

It will save the entropy values for different classes and different iterations in `Results/Entropy/Per_Class`


## Training Auxiliary Model
After storing the activation patterns, we need to run 

```
Auxiliary_Model.py
```

It will save the accuracy of the auxiliary model in `Results/Training_Testing_Custom_CNN_Accuracy.csv`

## Miscellaneous 

```
python Classification_Accuracy.py
```

Will store the classification accuracy of the dataset for iteration, as well as the classification accuracy of individual classes in `Accuracy` folder.

## CIFAR-10 ResNet-50
To run the experiments for the CIFAR-10 dataset on the ResNet-50 model, replace the files in `Code` with the file in `Code/CIFAR_10_ResNet`