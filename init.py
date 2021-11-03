from Perceptron import Perceptron
from Layer import Layer
from Multilayerperceptron import Multilayerperceptron

import pandas as pd
from pandas import read_csv
import random
#random.uniform(0, 1)

url = "/home/dark/Mywork/Multilayerperceptron/Data/trainingset.csv"
predictive_variables = read_csv(url)

url = "/home/dark/Mywork/Multilayerperceptron/Data/trainingpredictedvariable.csv"
predicted_variable = read_csv(url)

weight_matrice = [[random.uniform(0, 1)]*20,[random.uniform(0, 1)]*4]
first_layer_neurons = [None]*4
output_layer_neurons = [None]
input_layer = [None]*5

first_layer_neurons[0] = Perceptron(weight_matrice[0][0:5],input_layer,1)
first_layer_neurons[1] = Perceptron(weight_matrice[0][5:10],input_layer,1)
first_layer_neurons[2] = Perceptron(weight_matrice[0][10:15],input_layer,1)
first_layer_neurons[3] = Perceptron(weight_matrice[0][15:20],input_layer,1)
first_hidden_layer = Layer(first_layer_neurons,weight_matrice[0])


def backpropagation(epochs,learning_rate):
    epoch = 1
    while epoch <= epochs:
       
        for i in range(0,len(predictive_variables)):
             first_layer_neurons[0].inputs = predictive_variables.iloc[i].tolist()
             first_layer_neurons[1].inputs = predictive_variables.iloc[i].tolist()
             first_layer_neurons[2].inputs = predictive_variables.iloc[i].tolist()
             first_layer_neurons[3].inputs = predictive_variables.iloc[i].tolist()
             output_layer_neurons[0] =  Perceptron(weight_matrice[1],first_hidden_layer.compute_neurons_output(),0)
             output_layer = Layer(output_layer_neurons,weight_matrice[1])
             Layers = [first_hidden_layer,output_layer]
             Multilayer_perceptron = Multilayerperceptron(weight_matrice,Layers,5)
             y = Multilayer_perceptron.forward_pass(predictive_variables.iloc[i].tolist())
             for j in range(0,len(weight_matrice[1])):

                 weight_matrice[1][j] = weight_matrice[1][j] + learning_rate*((y[0]-predicted_variable.iloc[i].tolist()[0]))*first_layer_neurons[j].output
            

            

             for j in range(0,len(weight_matrice[0])):
                 if j<=4 and j>=0:
                    weight_matrice[0][j] = weight_matrice[0][j] + learning_rate*(y[0]-predicted_variable.iloc[i].tolist()[0])*weight_matrice[1][0]*predictive_variables.iloc[i].tolist()[j]*first_layer_neurons[0].output*(1-first_layer_neurons[0].output)
                 if j<=9 and j>=5:
                    weight_matrice[0][j] = weight_matrice[0][j] + learning_rate*(y[0]-predicted_variable.iloc[i].tolist()[0])*weight_matrice[1][1]*predictive_variables.iloc[i].tolist()[j-5]*first_layer_neurons[1].output*(1-first_layer_neurons[1].output)
                 if j<=14 and j>=10:
                    weight_matrice[0][j] = weight_matrice[0][j] + learning_rate*(y[0]-predicted_variable.iloc[i].tolist()[0])*weight_matrice[1][2]*predictive_variables.iloc[i].tolist()[j-10]*first_layer_neurons[2].output*(1-first_layer_neurons[2].output)
                 if j<=15 and j>=19:
                    weight_matrice[0][j] = weight_matrice[0][j] + learning_rate*(y[0]-predicted_variable.iloc[i].tolist()[0])*weight_matrice[1][3]*predictive_variables.iloc[i].tolist()[j-15]*first_layer_neurons[3].output*(1-first_layer_neurons[3].output)

             
                 
             first_layer_neurons[0].weights = weight_matrice[0][0:5]
             first_layer_neurons[1].weights = weight_matrice[0][5:10]
             first_layer_neurons[2].weights = weight_matrice[0][10:15]
             first_layer_neurons[3].weights = weight_matrice[0][15:20]
             output_layer_neurons[0].weights = weight_matrice[1]

        epoch = epoch + 1 
    return weight_matrice
#edit the number of epochs and learning rate in the argument of the below function
weight_matrice = backpropagation(100,1/len(predictive_variables))
S = 0
for i in range(0,len(predictive_variables)):
    first_layer_neurons[0].inputs = predictive_variables.iloc[i].tolist()
    first_layer_neurons[1].inputs = predictive_variables.iloc[i].tolist()
    first_layer_neurons[2].inputs = predictive_variables.iloc[i].tolist()
    first_layer_neurons[3].inputs = predictive_variables.iloc[i].tolist()
    output_layer_neurons[0] =  Perceptron(weight_matrice[1],first_hidden_layer.compute_neurons_output(),0)
    output_layer = Layer(output_layer_neurons,weight_matrice[1])
    Layers = [first_hidden_layer,output_layer]
    Multilayer_perceptron = Multilayerperceptron(weight_matrice,Layers,5)
    y = Multilayer_perceptron.forward_pass(predictive_variables.iloc[i].tolist())
    S = S + (y[0]-predicted_variable.iloc[i].tolist()[0])**2
#This program returns the value of the loss function (Average of Squared errors) at the end of the learning process, 
print(S/len(predictive_variables))









    