import numpy as np

# ----------------------------
# ReLU
# ----------------------------
class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0 # Replace 0 in dinputs where input values were negative or zero if(self.inputs <= 0): self.dinputs = 0 else: self.dinputs = dvalues

# ----------------------------
# Softmax
# ----------------------------
class Activation_Softmax:

    def forward(self, inputs):

        self.inputs = inputs

        exp_values = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )

        self.output = (
            exp_values /
            np.sum(exp_values, axis=1, keepdims=True)
        )