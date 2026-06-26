# Machine Learning Concepts

This document contains my notes on the fundamental concepts of Machine Learning while studying Artificial Intelligence from scratch.

---

# What is Machine Learning?

**Machine Learning (ML)** is a subfield of **Artificial Intelligence (AI)** in which computers learn patterns from data and improve their performance on a task without being explicitly programmed with rules for every possible situation.

Instead of writing fixed instructions, a machine learning model learns relationships from examples and uses them to make predictions or decisions.

---

# Types of Machine Learning

Machine Learning is generally divided into four categories.

## 1. Supervised Learning

The model is trained using **input data along with their correct labels (answers).**

Examples:

* House Price Prediction
* Spam Email Detection
* Handwritten Digit Classification

---

## 2. Unsupervised Learning

The model receives **only input data without labels** and attempts to discover hidden patterns or structures.

Examples:

* Customer Segmentation
* Clustering
* Dimensionality Reduction

---

## 3. Semi-Supervised Learning

A combination of supervised and unsupervised learning.

Only a small portion of the training data is labeled, while the remaining data is unlabeled.

Useful when labeling data is expensive or time-consuming.

---

## 4. Reinforcement Learning

In Reinforcement Learning (RL), an **agent** interacts with an **environment** by taking actions.

For each action, it receives:

* Rewards (good actions)
* Penalties (bad actions)

The objective is to maximize the total reward over time.

Examples:

* Robotics
* Game Playing
* Self-driving Cars

---

# Machine Learning Workflow

A typical Machine Learning project follows the steps below.

## 1. Problem Definition

Clearly define:

* What problem needs to be solved?
* What should the model predict?
* Why is the prediction valuable?

Examples:

* Predict customer churn
* Detect fraudulent transactions
* Recognize handwritten digits

---

## 2. Data Collection

Gather raw data from sources such as:

* Databases
* CSV files
* APIs
* Sensors
* Images
* User interactions

The quality and quantity of data greatly influence model performance.

---

## 3. Data Preprocessing

Prepare the data before training.

Common preprocessing steps include:

* Handling missing values
* Removing duplicate records
* Detecting outliers
* Feature scaling (Normalization / Standardization)
* Encoding categorical variables
* Feature Engineering

> **Garbage In, Garbage Out (GIGO):** Poor-quality data leads to poor model performance, regardless of the algorithm used.

---

## 4. Model Selection

Choose an appropriate algorithm based on the problem.

Examples include:

* Linear Regression
* Logistic Regression
* Decision Trees
* Random Forests
* Gradient Boosting
* Support Vector Machines
* Neural Networks

---

## 5. Model Training

Train the model using the training dataset.

During training:

* The model makes predictions.
* A loss function measures prediction error.
* An optimizer updates the model parameters to reduce the loss.

Common loss functions include:

* Mean Squared Error (MSE)
* Categorical Cross-Entropy
* Binary Cross-Entropy

---

## 6. Model Evaluation

Evaluate the trained model using validation or test data.

Common evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

## 7. Hyperparameter Tuning

Improve model performance by adjusting hyperparameters such as:

* Learning Rate
* Batch Size
* Number of Epochs
* Tree Depth
* Regularization Strength

Training and evaluation are repeated until satisfactory performance is achieved.

---

## 8. Deployment & Monitoring

Deploy the trained model into production through:

* Web APIs
* Mobile Applications
* Cloud Services
* Embedded Systems

After deployment, continuously monitor model performance and retrain the model when the data distribution changes.

---

# How Training Works

A machine learning model begins with randomly initialized parameters.

During each training iteration:

1. The model makes predictions.
2. The loss function measures prediction error.
3. Backpropagation computes gradients.
4. The optimizer updates the parameters.
5. The loss gradually decreases.

This process is repeated for many epochs until the model learns meaningful patterns from the data.

```
Initialize Parameters
        │
        ▼
Forward Pass
        │
        ▼
Compute Loss
        │
        ▼
Backpropagation
        │
        ▼
Update Parameters
        │
        ▼
Repeat Until Convergence
```

---

# Key Concepts to Learn

* Features
* Labels
* Training Data
* Validation Data
* Test Data
* Loss Function
* Optimizer
* Gradient Descent
* Overfitting
* Underfitting
* Bias and Variance
* Regularization
* Hyperparameters
