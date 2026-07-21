import numpy as np
import pandas as pd
from neural_network.core.layer import Layer_Dense
from neural_network.core.loss import Loss_CategoricalCrossentropy
from neural_network.core.activation import Activation_ReLU, Activation_Softmax
from neural_network.core.soft_loss import Activation_Softmax_Loss_CategoricalCrossentropy

np.random.seed(0)

class Convolution_Layer:
    def __init__(self, k, num_of_filters):
        self.num_of_filters = num_of_filters
        self.kernel = (
            np.random.randn(num_of_filters, k, k)
            * np.sqrt(2 / (k * k))
        ).astype(np.float32)
        self.biases = np.zeros(num_of_filters, dtype=np.float32)

    def convolve(self, patch, f):
        return np.sum(patch * self.kernel[f], axis=(1, 2)) + self.biases[f]
    
    def forward(self, batch):
        self.input = batch
        s, h, w = batch.shape
        k = self.kernel.shape[-1]

        self.output = np.zeros((s, (h-k)+1, (w-k)+1, self.num_of_filters), dtype=np.float32)

        for f in range(self.num_of_filters):
            for i in range(h-k+1):
                for j in range(w-k+1):
                    patch = batch[:, i:i+k, j:j+k]
                    self.output[:,i, j,f] = self.convolve(patch, f)
        return self.output
        
    def backward(self, dvalues):
        self.dbias = np.sum(dvalues, axis=(0,1,2))
        self.dkernel = np.zeros_like(self.kernel)
        self.dinputs = np.zeros_like(self.input)
        _, out_h, out_w, _ = dvalues.shape
        k = self.kernel.shape[-1]
        for f in range(self.num_of_filters):
            for i in range(out_h):
                for j in range(out_w):
                    patch = self.input[:,i:i+k, j:j+k]
                    gradient = dvalues[:, i, j,f][:, None, None]
                    self.dkernel[f] += np.sum(patch * gradient, axis=0)
                    self.dinputs[:, i:i+k, j:j+k] += (
                        gradient * self.kernel[f]
                    )
        return self.dinputs

class ReLU_Layer:
    def forward(self, batch):
        self.input = batch
        self.output = np.maximum(0,batch)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
        return self.dinputs

class MaxPooling_Layer:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, h, w, nof = inputs.shape
        p = self.pool_size
        out_h = h // p
        out_w = w // p

        self.output = np.zeros((batch_size, out_h, out_w, nof), dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):
                patch = inputs[
                    :,
                    i*p:(i+1)*p,
                    j*p:(j+1)*p,
                    :
                ]
                self.output[:,i, j,:] = np.max(patch, axis=(1, 2))
        return self.output
    
    def backward(self, dvalues):
        p = self.pool_size
        self.dinputs = np.zeros_like(self.inputs)

        s, out_h, out_w, channels = dvalues.shape

        for i in range(out_h):
            for j in range(out_w):
                patch = self.inputs[
                    :,
                    i*p:(i+1)*p,
                    j*p:(j+1)*p,
                    :
                ]

                for n in range(s):
                    for c in range(channels):
                        max_pos = np.unravel_index(
                            np.argmax(patch[n,:,:,c]),
                            (p, p)
                        )

                        self.dinputs[
                            n,
                            i*p + max_pos[0],
                            j*p + max_pos[1],
                            c
                        ] = dvalues[n, i, j,c]
        
        return self.dinputs

class Flatten_Layer:
    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.output = inputs.reshape(self.input_shape[0], -1)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs = dvalues.reshape(self.input_shape)
        return self.dinputs

# ----------------------------
# Load Data
# ----------------------------
scale_data = 1
data = np.array(
    pd.read_csv("C:\\Users\\HP\\Desktop\\AI-ML\\machineLearning\\data_sets\\digit_classifier_dataset_2.csv")
)

x = []
y = []

for row in data:
    label = int(row[0])
    x.append(np.array(row[1:]).reshape(64,64))

    l = np.zeros((1, 10))
    l[0, label] = 1
    y.append(l)

x = x*scale_data
y = y*scale_data

x = np.array(x)
y = np.array(y)

y = y.reshape(y.shape[0]*scale_data, -1)

x = x.astype(np.float32)

indices = np.random.permutation(len(x))

x = x[indices]
y = y[indices]

x_train = x[:4000*scale_data]
y_train = y[:4000*scale_data]

x_test = x[4000*scale_data:]
y_test = y[4000*scale_data:]

# ----------------------------
#           CNN Network
# ----------------------------

conv = Convolution_Layer(3, 4)
convReLU = ReLU_Layer()
pool = MaxPooling_Layer()
flatten = Flatten_Layer()

# ----------------------------
#   FULLY CONNECTED Network
# ----------------------------

dense1 = Layer_Dense(3844,16)
dense2 = Layer_Dense(16,10)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# ----------------------------
# Training Loop
# ----------------------------

epochs = 20
lr = 0.005

batch_size = 50

for epoch in range(epochs):
    #=========== Forward ===========

    #shuffling the the data set after each epoch
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    for start in range(0, len(x_train), batch_size):

        xb = x_train[start:start+batch_size]
        yb = y_train[start:start+batch_size]

        conv.forward(xb)

        convReLU.forward(conv.output)

        pool.forward(convReLU.output)

        flatten.forward(pool.output)

        dense1.forward(flatten.output)

        activation1.forward(dense1.output)

        dense2.forward(activation1.output)

        activation2.forward(dense2.output)

        #=========== Loss & Accuracy calculation ===========
        loss = loss_function.calculate(activation2.output, yb)

        predictions = np.argmax(activation2.output, axis=1)

        y_true = np.argmax(yb, axis=1)

        accuracy = np.mean(predictions == y_true)

        #=========== Backward ===========

        loss_activation.backward(activation2.output, yb)

        dense2.backward(loss_activation.dinputs)

        activation1.backward(dense2.dinputs)

        dense1.backward(activation1.dinputs)

        flatten.backward(dense1.dinputs)

        pool.backward(flatten.dinputs)

        convReLU.backward(pool.dinputs)

        conv.backward(convReLU.dinputs)

        #=========== Update ============

        dense1.weights += (-lr * dense1.dweights)

        dense1.biases += (-lr * dense1.dbiases)

        dense2.weights += (-lr * dense2.dweights)

        dense2.biases += (-lr * dense2.dbiases)
    
        conv.kernel += (-lr * conv.dkernel)

        conv.biases += (-lr * conv.dbias)

    if True or epoch % 200 == 0:
        print(
            f"Epoch {epoch} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )



# ----------------------------
# Saving the Model
# ----------------------------

# uncomment the section below if want to update the weights of the model

np.savez(
    r"C:\Users\HP\Desktop\AI-ML\machineLearning\neural_network\models\cnn_digit_classifier_batch_processing_model.npz",
    dense1_weights=dense1.weights,
    dense1_biases=dense1.biases,
    dense2_weights=dense2.weights,
    dense2_biases=dense2.biases,
    conv_biases = conv.biases,
    conv_kernel = conv.kernel
)
print("Model Saved!")


conv_out = conv.forward(x_test)
relu_out = convReLU.forward(conv_out)
pool_out = pool.forward(relu_out)

flat_out = flatten.forward(pool_out)

dense1.forward(flat_out)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

prediction = np.argmax(activation2.output,axis=1)

confidence = np.max(activation2.output)

print("\nPredictions:")
print(prediction)
print("\nActual:")
print(np.argmax(y_test, axis=1))