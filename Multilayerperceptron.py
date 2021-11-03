from Layer import Layer


class Multilayerperceptron:
    weight_matrice = [[None]]
    Layers = [None]
    Input_layer = [None]
    

    def __init__(self,weight_matrice,Layers,Input_layer_length):
        self.Input_layer = [None]*Input_layer_length
        self.weight_matrice = weight_matrice
        self.Layers = Layers
        
    
    def forward_pass(self,Input_layer):
        self.Input_layer = Input_layer
        layer_output = []
        for x in self.Layers[0].Neurons:
            x.inputs = Input_layer
        for layer in self.Layers:
            layer_output = layer_output + [layer.compute_neurons_output()]
        return layer_output[len(self.Layers)-1]
        


