import math
class Perceptron:
    weights = None
    inputs = None
    output = None
    activation_function_number = None
    def __init__(self,weights,inputs,activation_function_number) :
        self.weights = weights
        self.inputs = inputs
        self.activation_function_number = activation_function_number
    

    def activation_function(self,x):
        if self.activation_function_number == 0:
            return x
        if self.activation_function_number == 1:
            return 1/(1+math.exp(-x))

    
    def compute_output(self):
        weighted_sum = 0
        for i in range(0,len(self.weights)):
            weighted_sum = weighted_sum + self.weights[i]*self.inputs[i]
        
        self.output = self.activation_function(weighted_sum)
        return self.output


