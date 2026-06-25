import numpy as np
from nnfs.datasets import spiral_data

np.random.seed(0)

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot( self.inputs.T, dvalues )
        self.dbiases = np.sum( dvalues, axis=0, keepdims=True )
        self.dinputs = np.dot( dvalues, self.weights.T )

class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0 # Replace 0 in dinputs where input values were negative or zero if(self.inputs <= 0): self.dinputs = 0 else: self.dinputs = dvalues


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

class Loss:

    def calculate(self, output, y):

        sample_losses = self.forward(output, y)

        return np.mean(sample_losses)

class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred_clipped = np.clip(
            y_pred,
            1e-7,
            1 - 1e-7
        )

        if len(y_true.shape) == 1:

            correct_confidences = (
                y_pred_clipped[
                    range(samples),
                    y_true
                ]
            )

        else:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = (
            -np.log(correct_confidences)
        )

        return negative_log_likelihoods
    
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(
                y_true,
                axis=1
            )

        self.dinputs = dvalues.copy()

        self.dinputs[
            range(samples),
            y_true
        ] -= 1

        self.dinputs = (
            self.dinputs / samples
        )


x, y = spiral_data(samples = 100, classes = 3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2_sm = Activation_Softmax()

dense1.forward(x)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2_sm.forward(dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2_sm.output, y)

print("Loss:", loss)