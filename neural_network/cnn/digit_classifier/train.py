import numpy as np
import pandas as pd
from neural_network.cnn.cnn import(
    Convolution_Layer,
    ReLU_Layer,
    MaxPooling_Layer,
    Flatten_Layer
)
from neural_network.core.layer import Layer_Dense
from neural_network.core.loss import Loss_CategoricalCrossentropy
from neural_network.core.activation import Activation_ReLU, Activation_Softmax
from neural_network.core.soft_loss import Activation_Softmax_Loss_CategoricalCrossentropy

np.random.seed(123)

# ----------------------------
# Load Data
# ----------------------------

data = np.array(
    pd.read_csv("C:\\Users\\HP\\Desktop\\AI-ML\\machineLearning\\data_sets\\digit_classifier_dataset.csv")
)

x = []
y = []

for row in data:
    label = int(row[0])
    x.append(row[1:])

    l = np.zeros((1, 10))
    l[0, label] = 1
    y.append(l)

x = np.array(x)
y = np.array(y)

x = x.astype(np.float32)
x = np.nan_to_num(x, nan=1.0)

image=(x[0]).reshape(64,64)
label=(y[0])

epochs = 10
lr = 0.001

# ----------------------------
#           CNN Network
# ----------------------------

conv = Convolution_Layer(3)
convReLU = ReLU_Layer()
pool = MaxPooling_Layer()
flatten = Flatten_Layer()

conv_out = conv.forward(image)
relu_out = convReLU.forward(conv_out)
pool_out = pool.forward(relu_out)
flat_out = flatten.forward(pool_out)
flat_out = flat_out.reshape(1, -1) ###

# ----------------------------
#   FULLY CONNECTED Network
# ----------------------------

dense1 = Layer_Dense(flat_out.shape[1],16)
dense2 = Layer_Dense(16,10)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

dense1.forward(flat_out)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

for epoch in range(epochs):
    if epoch%10==0:
        print("*", end="")
    total_loss = []
    total_accuracy = []
    # ----------------------------
    # Training Loop
    # ----------------------------
    for i in range(len(x)):
        #=========== Loss & Accuracy calculation ===========
        loss = loss_function.calculate(activation2.output, label)

        predictions = np.argmax(activation2.output, axis=1)
    
        y_true = np.argmax(label, axis=1)
    
        accuracy = np.mean(predictions == y_true)

        if True:
            total_loss.append(loss)
            total_accuracy.append(accuracy)
    
        #=========== Backward ===========
    
        loss_activation.backward(activation2.output, label)
    
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
    
        #=========== Forward ===========

        image=(x[i]).reshape(64,64)
        label=(y[i])
    
        conv.forward(image)
        
        convReLU.forward(conv.output)
        
        pool.forward(convReLU.output)
        
        flatten.forward(pool.output)
    
        flatten.output = flatten.output.reshape(1, -1) ###
    
        dense1.forward(flatten.output)
    
        activation1.forward(dense1.output)
    
        dense2.forward(activation1.output)
    
        activation2.forward(dense2.output)

    if epoch%200==0:
        print(
            f"\nEpoch {epoch} | "
            f"Loss: {np.mean(total_loss):.4f} | "
            f"Accuracy: {np.mean(total_accuracy):.4f}\n"
        )


# ----------------------------
# Saving the Model
# ----------------------------

# uncomment the section below if want to update the weights of the model

np.savez(
    r"C:\Users\HP\Desktop\AI-ML\machineLearning\neural_network\models\cnn_digit_classifier_model.npz",
    dense1_weights=dense1.weights,
    dense1_biases=dense1.biases,
    dense2_weights=dense2.weights,
    dense2_biases=dense2.biases,
    conv_biases = conv.biases,
    conv_kernel = conv.kernel
)
print("Model saved!")


image=(x[509]).reshape(64,64)
label=(y[509])

conv_out = conv.forward(image)
relu_out = convReLU.forward(conv_out)
pool_out = pool.forward(relu_out)
flat_out = flatten.forward(pool_out)
flat_out = flat_out.reshape(1, -1)

dense1.forward(flat_out)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

prediction = np.argmax(activation2.output,axis=1)

confidence = np.max(activation2.output)

print("\nPredictions:")
print(prediction)
print("\nActual:")
print(np.argmax(label, axis=1))