import numpy as np
np.random.seed(0)

class Flatten_Layer:
    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.output = inputs.reshape(self.input_shape[0], -1)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs = dvalues.reshape(self.input_shape)
        return self.dinputs
