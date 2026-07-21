import numpy as np

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

MODEL_PATH = r"C:\Users\HP\Desktop\AI-ML\machineLearning\neural_network\models\cnn_digit_classifier_model.npz"
saved = np.load(MODEL_PATH)

# load image
image = None

conv = Convolution_Layer(3)
convReLU = ReLU_Layer()
pool = MaxPooling_Layer()
flatten = Flatten_Layer()

conv.kernel = saved["conv_kernel"]
conv.biases = saved["conv_biases"]

conv_out = conv.forward(image)
relu_out = convReLU.forward(conv_out)
pool_out = pool.forward(relu_out)
flat_out = flatten.forward(pool_out)
flat_out = flat_out.reshape(1, -1) ###

dense1 = Layer_Dense(flat_out.shape[1],16)
dense2 = Layer_Dense(16,10)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

dense1.weights = saved["dense1_weights"]
dense1.biases = saved["dense1_biases"]
dense2.weights = saved["dense2_weights"]
dense2.biases = saved["dense2_biases"]

dense1.forward(flat_out)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)