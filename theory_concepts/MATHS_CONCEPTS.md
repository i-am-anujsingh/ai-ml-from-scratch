## Mathematics Concepts

This document contains some of the fundamental concepts I learned while exploring AI/ML.

---

## Matrices and Rank

A matrix is simply a rectangular arrangement of numbers.
Rank = Amount of unique information (row in matrices).
A matrix can never have a rank greater than its smaller dimension.

For example:

A 3 × 5 matrix
* Rows = 3
* Columns = 5

Maximum rank = 3

A 7 × 2 matrix

Maximum rank = 2

In general, **Rank≤min(rows,columns)**

---

## The Special Vectors

Some vectors are special, after applying the matrix:

Their direction does not change.
Only their length changes (they may also flip direction if multiplied by a negative number).

These are called **eigenvectors**.

Example:

Consider

A=[[ 2 0 ]
   [ 0 3 ]]

Take

v=[[1]
   [0]]

Then

Av=[[2]
    [0]] = 2[[1]
             [0]]

Notice:
Direction stayed the same.
Length became twice as large.

v is an eigenvector.

The number 2 is its **eigenvalue**.

### Key Points to Remember
* A matrix transforms vectors.
* Most vectors change direction after the transformation.
* Eigenvectors keep their direction.
* Eigenvalues tell you how much those eigenvectors are stretched, compressed, or flipped.
* The fundamental equation is: Av=λv, λ is eigenvalue and v is eigenvector.
* Eigenvalues and eigenvectors are used in PCA, image compression, graph algorithms, recommendation systems, and many other areas of AI.

---

## Gradient

The gradient is simply the collection of all partial derivatives.

The gradient tells us:
Which direction increases the function the fastest.

Gradient Descent updates the weights using:-

Wnew = Wold − η∇L

where:

* W = weights
* η (eta) = learning rate
* ∇L = gradient

∇L = gradient

The minus sign means we move **opposite** the gradient because we want to reduce the loss.

