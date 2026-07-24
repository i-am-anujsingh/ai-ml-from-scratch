import numpy as np
import pandas as pd

from neural_network.core.dense_layer import Layer_Dense
from neural_network.core.loss import Loss_CategoricalCrossentropy
from neural_network.core.activation import Activation_ReLU, Activation_Softmax
from neural_network.core.soft_loss import Activation_Softmax_Loss_CategoricalCrossentropy

from neural_network.cnn.core.batch_processing_layers.flatten import Flatten_Layer
from neural_network.cnn.core.batch_processing_layers.maxpool import MaxPooling_Layer
from neural_network.cnn.core.batch_processing_layers.convolution import Convolution_Layer, FastConvolution_Layer

np.random.seed(0)

# ----------------------------
# Load Data
# ----------------------------
scale_data = 1
data = np.array(
    pd.read_csv("C:\\Users\\HP\\Desktop\\AI-ML\\machineLearning\\data_sets\\cnn_dataset\\digit_classifier_dataset_2.csv")
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

conv = FastConvolution_Layer(3, 4)
convReLU = Activation_ReLU()
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
    
    epoch_loss = 0
    epoch_correct = 0
    epoch_samples = 0

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

        epoch_loss += loss * xb.shape[0]

        epoch_correct += (predictions == y_true).sum()

        epoch_samples += xb.shape[0]

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
        print(f"Epoch {epoch} | Loss: {epoch_loss/epoch_samples:.4f} | Acc: {epoch_correct/epoch_samples:.4f}")


# ----------------------------
# Saving the Model
# ----------------------------

# uncomment the section below if want to update the weights of the model

np.savez(
    r"C:\Users\HP\Desktop\AI-ML\machineLearning\neural_network\models\cnn_digit_classifier_batch_processing_model_test.npz",
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