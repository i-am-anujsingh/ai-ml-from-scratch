# Handwritten Digit Recognition

A handwritten digit recognition system implemented entirely from scratch using **Python** and **NumPy** without relying on deep learning frameworks such as TensorFlow or PyTorch.

## Overview

This project was built to understand the internal working of neural networks by implementing every major component manually.

The model classifies handwritten digits from **0–9** using a feedforward neural network trained on a custom dataset.

## Features

* Dense Layers
* ReLU Activation
* Softmax Output Layer
* Cross-Entropy Loss
* Backpropagation
* Gradient Descent
* Model Serialization (.npz)

## Dataset

Instead of using the MNIST dataset, I created a custom handwritten dataset and expanded it using image preprocessing techniques.

The model was trained on approximately **100–200 augmented images per digit**.

## Current Limitations

* Performs best on handwriting similar to the training data.
* Limited dataset size reduces generalization.
* Accuracy decreases on unseen writing styles and inverted color schemes.

## Future Improvements

* Larger dataset
* Data augmentation
* CNN implementation
* Better weight initialization
* Batch normalization

## Learning Outcomes

This project helped me understand:

* Forward Propagation
* Backpropagation
* Gradient Descent
* Weight Initialization
* Activation Functions
* Neural Network Training Pipeline

This project is part of my AI From Scratch learning journey.
