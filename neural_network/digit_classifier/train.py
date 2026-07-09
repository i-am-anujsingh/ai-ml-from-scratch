'''
Training file of the digit recognition neural network
'''
import numpy as np
import pandas as pd
from core.layer import Layer_Dense
from core.loss import Loss_CategoricalCrossentropy
from core.activation import Activation_ReLU, Activation_Softmax
from core.soft_loss import Activation_Softmax_Loss_CategoricalCrossentropy

np.random.seed(0)

learning_rate = 0.013

epochs = 1000

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

# make batch
y = y.reshape(2010, -1)

# spliting data set
x_train = x[:-10]
y_train = y[:-10]

x_test = x[-10:]
y_test = y[-10:]

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# ----------------------------
#           Network
# ----------------------------
dense1 = Layer_Dense(
    x.shape[1],
    64
)

dense2 = Layer_Dense(
    64,
    10
)


activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

loss_function = (
    Loss_CategoricalCrossentropy()
)

loss_activation = (
    Activation_Softmax_Loss_CategoricalCrossentropy()
)


# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(epochs):

#=========== Forward ===========

    dense1.forward(x_train)

    activation1.forward(
        dense1.output
    )

    dense2.forward(
        activation1.output
    )

    activation2.forward(
        dense2.output
    )

    loss = loss_function.calculate(
        activation2.output,
        y_train
    )

    # Converting predictions from one-hot encoded format to class indices / sparse labels
    predictions = np.argmax(
        activation2.output,
        axis=1
    )

    # Converting one-hot encoded labels back to class indices / sparse labels
    y_true = np.argmax(
        y_train,
        axis=1
    )
    # Calculating accuracy by comparing predicted class indices with true class indices returns 1  if they match and 0 if they don't, then taking the mean gives the overall accuracy
    accuracy = np.mean(
        predictions == y_true
    )

#=========== Backward ===========

    loss_activation.backward(
        #predictions in one-hot encoded format
        activation2.output,
        #actual labels in one-hot encoded format
        y_train
    )

    dense2.backward(
        loss_activation.dinputs
    )

    activation1.backward(
        dense2.dinputs
    )

    dense1.backward(
        activation1.dinputs
    )

#=========== Update ============

    dense1.weights += (
        -learning_rate *
        dense1.dweights
    )

    dense1.biases += (
        -learning_rate *
        dense1.dbiases
    )

    dense2.weights += (
        -learning_rate *
        dense2.dweights
    )

    dense2.biases += (
        -learning_rate *
        dense2.dbiases
    )

    if epoch % 200 == 0:

        print(
            f"Epoch {epoch} | "
            f"Loss: {loss:.6f} | "
            f"Accuracy: {accuracy:.4f}"
        )

# ----------------------------
# Saving the Model
# ----------------------------

# uncomment the section below if want to update the weights of the model
'''
np.savez(
    "./digit_model.npz",
    dense1_weights=dense1.weights,
    dense1_biases=dense1.biases,
    dense2_weights=dense2.weights,
    dense2_biases=dense2.biases
)
print("Model saved!")'''

# ----------------------------
# Final Prediction
# ----------------------------
dense1.forward(x_test)

activation1.forward(
    dense1.output
)

dense2.forward(
    activation1.output
)

activation2.forward(
    dense2.output
)

prediction = np.argmax(
    activation2.output,
    axis=1
)

confidence = np.max(
    activation2.output
)


print("\nPredictions:")
print(prediction)
print("\nActual:")
print(np.argmax(y_test, axis=1))