The data used for simulation was extracted from : 
https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
Before we dig into explanationn we would like to clarify some notations :
Wij is the weight between input node 𝑖 in the input layer and the hidden node 𝑗 in the hidden layer.
Yj is the neuron ouput indexed by j.
Xi is the ith element of the feature vector. 
Networks architecture : 

1) Input Layers : 5 length vector of inputs described in the link above.

Hidden layer : 4 neurons equiped with the sigmoid activation function f(x) = 1/(1+exp(-x)). The inputs are : Xi for i in [1,2,3,4,5]. The weighted sum of inputs is of the following form : Zj = sumi=1->5 Wij*Xi. the output is of the form : Yj = f(Zj)

Ouput layer : 1 neurons equiped with the identity activation function g(x) = x. We have only one neuron in the output layer, we will index it using "k".The inputs are Yj for j in [1,2,3,4].The weighted sum of inputs is Zk = sumi=1->4 Wik*Yi, The final output is Yk = g(Zk)
Loss function : L = (sumi=1->n (Y(predicted) - Y(true value))*(Y(predicted) - Y(true value)))/n

2) Learning type : Online learning ( weight are constantly updated after each forward pass )


3) algorithm 
1. Initialize all weights to small random values.
2. Standardize training data.
3. For each input:
 . Calculate the output (forward pass);
 . Update the weights by backpropagating using the follow role:
   Wij = Wij + Learning_rate*(dL/dWjk)

   dL/dWjk = (Yk-Tk)*Yj              (k refers to the neuron in the ouput layer, j refers to one of the neurons in the hidden layer)
   dL/dWij = (Yk-Tk)*Yj*(1-Yj)*Xi*Wjk              (i refers to an element in the input layer, j refers to one of the neurons in the hidden layer)
4) repeat step 3 until a stopping criteria is met


The folder "Data" contains all the training data (trainingset.csv) and test data (testset.csv),the data was scaled for both configs. The target variable for both training and testing sets (trainingpredictedvariable.csv) and (testpredictedvariable.csv) are stored seperately in another file. The "airfoil_self_noise.csv" shows the raw form of data, "processed_data.csv" shows the scaled data, the rest of the file presents a subset of data (training and testing). 

The implementation is done from scratch, therefore you will need to rebuild everything or just adapt it to your specific task. You will need to handle everything manually, therefore a clear understanding of Feed-Forward neural networks is required in order to avoid concept-related mistakes.



run the file init.py, the program will come up with the empirical error after the learning process, don't hesitate to control the epoch parameter, and learning rate.
