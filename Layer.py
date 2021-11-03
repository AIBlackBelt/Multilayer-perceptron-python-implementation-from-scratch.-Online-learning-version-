class Layer:
    Neurons = [None]
    Length = 0 
    input_weights = [None]
    Neurons_output = []

    def __init__(self,Neurons,input_weights):
        self.Neurons = Neurons
        self.Length = len(Neurons)
        self.input_weights = input_weights
    
    

    def compute_neurons_output(self):
        for neuron in self.Neurons:
            self.Neurons_output = self.Neurons_output + [neuron.compute_output()]
        return self.Neurons_output

       

