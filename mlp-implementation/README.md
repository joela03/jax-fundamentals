# Multi-Layer Perceptron in JAX for Fashion MNIST

A pure JAX implementation of a neural network classifier demonstrating fundamental concepts from Michael Nielsen's Neural Networks and Deep Learning (Chapters 1-2).

## Overview

This project implements a feedforward neural network from scratch using JAX to classify Fashion MNIST images (10 clothing categories). The implementation emphasises understanding core concepts while leveraging JAX's modern approach to numerical computing.

**Final Performance:** 85.9% test accuracy

## Architecture
```
Input Layer:    784 neurons (28×28 flattened grayscale images)
Hidden Layer 1: 128 neurons (ReLU activation)
Hidden Layer 2: 64 neurons (ReLU activation)
Output Layer:   10 neurons (Softmax activation)
```

**Total Parameters:** 109,386

## Core Concepts Implemented

### 1. Feedforward Architecture
The network processes input through sequential layers, each applying an affine transformation followed by a non-linear activation:
```
z^l = W^l · a^(l-1) + b^l
a^l = σ(z^l)
```

### 2. Backpropagation via Automatic Differentiation
Rather than manually computing gradients using the chain rule, this implementation uses JAX's `grad()` function, which automatically computes derivatives through reverse-mode autodiff. This is both more efficient and less error-prone than manual backprop.

### 3. Stochastic Gradient Descent (SGD)
Parameters are updated using mini-batch gradient descent:
```
θ_new = θ_old - η∇L(θ)
```

where η is the learning rate and ∇L is the gradient of the loss function.

## Key Design Decisions

### Weight Initialisation: Scaled Gaussian
```python
W ~ N(0, 1/√n_in)
```

**Why not simple N(0,1)?**
- With 784 inputs, variance would explode to 784
- Large activations → saturated softmax → vanishing gradients
- Scaling by 1/√n_in preserves variance across √layers (Var ≈ 1)
- This enables stable training and proper gradient flow

**Mathematical reasoning:**
- For z = w₁x₁ + w₂x₂ + ... + wₙxₙ
- If Var(wᵢ) = 1/n and Var(xᵢ) = 1
- Then Var(z) = n × (1/n × 1) = 1 ✓

### Activation Functions

**ReLU for Hidden Layers:**
```python
f(x) = max(0, x)
```

- Simple and efficient
- Mitigates vanishing gradient problem
- Introduces necessary non-linearity

**Softmax for Output:**
```python
σ(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

- Converts logits to probability distribution
- Pairs naturally with cross-entropy loss
- Differentiable for backpropagation

### Loss Function: Cross-Entropy
```python
L = -Σ y_true · log(y_pred)
```

**Why not Mean Squared Error?**
- Cross-entropy has better gradient properties (no learning slowdown)
- Natural interpretation with softmax (maximises log-likelihood)
- MSE can cause learning to slow when the network is very wrong

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 0.01 | Balanced between speed and stability |
| Batch Size | 128 | Good trade-off between gradient accuracy and speed |
| Epochs | 20 | 10 is sufficient but 20 allows more time to learn and improve scores |

## JAX-Specific Features

### 1. Pure Functional Programming
- Parameters are explicit data structures, not hidden in objects
- No side effects or in-place operations
- Functions are composable and transformable

### 2. JIT Compilation
```python
@jit
def train_step(params, x_batch, y_batch, learning_rate):
    # Compiled by XLA for 10-100x speedup
```

### 3. Functional Random Number Generation
JAX requires explicit PRNG key management:
```python
key, subkey = jax.random.split(key)  # Explicit key splitting
batches = create_batches(X, y, batch_size, subkey)
```

This ensures reproducibility and enables parallelisation.

### 4. Automatic Differentiation
```python
grads = grad(loss_fn)(params, x_batch, y_batch)
```

JAX computes exact gradients using reverse-mode autodiff, implementing backpropagation automatically.

## Performance Characteristics

**Training Time (CPU):** ~5 seconds for 10 epochs
- First epoch: ~1.4s (JIT compilation)
- Subsequent epochs: ~0.3s each

**Memory:** Minimal (~100k parameters)

## Results
```
Final Test Accuracy:  85.9%
Final Training Loss:  ~0.5
```

Fashion MNIST is more challenging than digit MNIST due to greater intra-class variation. For comparison:
- Random guessing: 10%
- Simple MLP (this implementation): ~86%
- State-of-the-art CNNs: ~95%

## What This Demonstrates

1. **Understanding of Neural Network Fundamentals**
   - Forward/backward propagation
   - Gradient descent optimisation
   - Loss functions and activation functions

2. **JAX Proficiency**
   - Pure functional programming paradigm
   - Automatic differentiation
   - JIT compilation
   - Proper PRNG key management

3. **Mathematical Insight**
   - Why variance preservation matters
   - How activation functions affect learning
   - Trade-offs in optimisation

## Running the Code
```bash
pip install -r requirements.txt

python mlp_fashion_mnist.py
```

## Future Improvements

Potential enhancements to explore:
- Learning rate scheduling (decay over time)
- Momentum-based optimisation (SGD with momentum, Adam)
- Regularisation (L2, dropout)
- Deeper architectures
- Convolutional layers for spatial structure

## References

- Nielsen, M. (2015). Neural Networks and Deep Learning. Chapters 1-2.
- JAX Documentation: https://jax.readthedocs.io/
- Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist

---

**Author:** Joël Allen-Caliste 
**Purpose:** Understanding neural network fundamentals through implementation  
**Date:** 23rd December 2025