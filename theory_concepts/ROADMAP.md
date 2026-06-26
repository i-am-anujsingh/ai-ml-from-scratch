# AI Learning Roadmap

This roadmap outlines my learning journey in Artificial Intelligence, from Machine Learning fundamentals to Neural Networks and Large Language Models.

## Learning Path

```text
Artificial Intelligence (AI)
│
├── Machine Learning (ML)
│   │
│   └── Deep Learning (DL)
│       │
│       └── Neural Networks
│
└── Large Language Models (LLMs)
```

## Topics Covered

### Machine Learning

* Python for Machine Learning
* NumPy
* Pandas
* Data Visualization
* Linear Regression
* Logistic Regression
* Train/Test Split
* Overfitting & Underfitting
* Gradient Descent

### Neural Networks

* Dense Layers
* Activation Functions
* Forward Propagation
* Backpropagation
* Loss Functions
* Gradient Descent
* Model Evaluation

### Large Language Models (Current)

* Tokenization
* Embeddings
* Self-Attention
* Multi-Head Attention
* Transformer Architecture
* GPT Models

## Neural Network Training Pipeline

```
                                         _______________
                                        |               |             __
                                        |    Update     |               |
                    ____________________|    weights    |               |
                    |                   |               |               |
                    |                   |               |               |     _______________
                    |                   |_______________|               |    |               |
                    ↓                                ↑                  \   |               |
            _______________                          |                    \  |    COMPLETE   |
            |              |                         |                    /  |      MODE     |
            |    MODEL     |                         |                   /   |               |
features--->| (prdiction   |                    _____|_________         |    |_______________|
            |  function)   |------------------→ |              |        |
            |______________|                    |   Compute    |        |
                                                |     Loss     |        |
                                                |              |      __|
labels ---------------------------------------→ |______________|  


                                   [ OR ]


  Input Features and Labels
             │
             ↓
       ┌─────────────────┐
       │ Forward Pass    │
       └─────────────────┘
             │
             ↓
       ┌─────────────────┐
       │ Predictions     │
       └─────────────────┘
             │
             ↓
       ┌─────────────────┐
       │ Compute Loss    │
       └─────────────────┘
             │
             ↓
       ┌─────────────────┐
       │ Backpropagation │
       └─────────────────┘
             │
             ↓
       ┌─────────────────┐
       │ Update Weights  │
       └─────────────────┘
             │
             └───────────────┐
                             ↓
                      NEXT ITERATION
```

## Current Focus

* Building Neural Networks from Scratch
* Understanding Transformer Architecture
* Implementing a GPT-style Language Model
* Strengthening mathematical foundations behind AI
