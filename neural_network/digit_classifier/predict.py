'''
model file of the digit recognition neural network
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

from core.layer import Layer_Dense
from core.activation import Activation_ReLU, Activation_Softmax

# ----------------------------
# Load Data
# ----------------------------
data = []
x = []
path = input("Enter the path to the image of number recognition: ")
img = plt.imread(path)
img = cv2.resize(img, (32, 32)) # resizing image to 32x32 according to the training data
flat = img.reshape(-1)
data.append(flat)
for row in data:
    x.append(row)
x = np.array(x)
x = x.astype(np.float32)
x = np.nan_to_num(x, nan=1.0)

# ----------------------------
#           Network
# ----------------------------
dense1 = Layer_Dense(x.shape[1], 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 10)
activation2 = Activation_Softmax()

# ----------------------------
# Loading the Model
# ----------------------------
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "digit_model.npz"
saved = np.load(MODEL_PATH)
dense1.weights = saved["dense1_weights"]
dense1.biases = saved["dense1_biases"]
dense2.weights = saved["dense2_weights"]
dense2.biases = saved["dense2_biases"]

# ----------------------------
# Final Prediction
# ----------------------------
dense1.forward(x)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
prediction = np.argmax(activation2.output,axis=1)

print("\nPredictions: ", prediction[0])