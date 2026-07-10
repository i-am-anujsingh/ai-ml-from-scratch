# Neural Network Concepts

This document contains the fundamental concepts I learned while implementing a feedforward neural network from scratch using Python and NumPy.

---

# Neural Network Architecture

```
Input
  │
  ▼
Dense Layer
  │
  ▼
ReLU
  │
  ▼
Dense Layer
  │
  ▼
ReLU
  │
  ▼
Dense Layer
  │
  ▼
Softmax
  │
  ▼
Probabilities
```

---

# Dense Layer (Fully Connected Layer)

A **Dense Layer** (also called a **Fully Connected Layer**) is a neural network layer in which every input is connected to every neuron.

It performs the following operation:

```
Z = X · W + B
```

where:

* **X** → Input Features
* **W** → Weights
* **B** → Biases
* **Z** → Output of the Layer

This is the fundamental computation performed in most neural networks.

## Why Matrix Multiplication?

Neural networks use **matrix multiplication** because it allows the computation of outputs for all neurons simultaneously.

Instead of calculating each neuron individually, vectorization enables efficient computation, making training significantly faster.

---

# Activation Functions

An **Activation Function** determines whether a neuron should activate and what value it should pass to the next layer.

Without activation functions, a neural network becomes only a series of linear transformations and cannot learn complex patterns.

Activation functions introduce **non-linearity**, enabling neural networks to model complicated relationships in data.

---

## Types of Activation Functions

* Linear Activation Functions
* Non-Linear Activation Functions

Modern neural networks primarily use **non-linear activation functions**.

---

## Common Activation Functions

| Function    | Common Usage               |
| ----------- | -------------------------- |
| Binary Step | Early Perceptrons          |
| Linear      | Regression Output Layers   |
| Sigmoid     | Binary Classification      |
| Tanh        | Older Hidden Layers        |
| ReLU        | Modern Hidden Layers       |
| Leaky ReLU  | Improved ReLU              |
| ELU         | Deep Neural Networks       |
| Softmax     | Multi-class Classification |

---

# ReLU (Rectified Linear Unit)

ReLU is the most commonly used activation function for hidden layers.

```
f(x) = max(0, x)
```

Properties:

* Negative values become **0**
* Positive values remain unchanged
* Computationally efficient
* Helps reduce the Vanishing Gradient problem

---

## Dying ReLU Problem

If a neuron continuously receives negative inputs:

* Output becomes **0**
* Gradient becomes **0**
* The neuron stops learning permanently

This issue is known as the **Dying ReLU Problem**.

---

# Leaky ReLU

Leaky ReLU addresses the Dying ReLU problem by allowing a small gradient for negative values.

```
f(x) = max(0.01x, x)
```

Instead of outputting zero for negative inputs, it outputs a small negative value.

---

# Softmax Activation Function

Softmax is typically used in the **output layer** for multi-class classification.

It converts raw outputs (**logits**) into probabilities.

Example:

Raw Output:

```
[2.1, 1.3, 0.2]
```

After Softmax:

```
[0.62, 0.28, 0.10]
```

Properties:

* Values lie between **0 and 1**
* Sum of all probabilities equals **1**
* Highest probability corresponds to the predicted class

---

# Categorical Cross-Entropy (CCE)

Categorical Cross-Entropy is the standard loss function for multi-class classification.

It measures how different the predicted probability distribution is from the true distribution.

* Correct predictions → Low Loss
* Incorrect predictions → High Loss

### Example

Classes:

| Class | Label |
| ----- | ----- |
| 0     | Cat   |
| 1     | Dog   |
| 2     | Bird  |

Suppose the correct class is **Dog**.

The one-hot encoded target becomes:

```
Cat   Dog   Bird
0      1      0
```

The model compares its predicted probabilities with this target to compute the loss.

---

# Vanishing Gradient Problem

During backpropagation, gradients are propagated from the output layer toward earlier layers.

If these gradients become extremely small, earlier layers receive almost no learning signal.

As a result:

* Weight updates become negligible.
* Learning slows dramatically.
* Very deep networks become difficult to train.

---

# Xavier (Glorot) Initialization

Developed by **Xavier Glorot**.

The goal is to initialize weights so that activations neither explode nor vanish as they propagate through the network.

For a layer with:

* **n_inputs** → Number of input neurons
* **n_outputs** → Number of output neurons

Weight variance:

```
Var(W) = 2 / (n_inputs + n_outputs)
```

Commonly used with:

* Sigmoid
* Tanh

---

# He Initialization

Developed by **Kaiming He**.

Since ReLU sets negative values to zero, Xavier initialization is not always ideal.

He Initialization compensates for this behavior.

Weight variance:

```
Var(W) = 2 / n_inputs
```

Commonly used with:

* ReLU
* Leaky ReLU
