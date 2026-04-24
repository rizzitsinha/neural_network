# Neural Network — NumPy vs TensorFlow/Keras Benchmark

A from-scratch implementation of a fully-connected neural network in pure NumPy, trained on MNIST and benchmarked against an equivalent TensorFlow/Keras model.

---

## Overview

This project implements a multi-layer perceptron (MLP) from the ground up using only NumPy — no deep learning frameworks. It covers the full training pipeline: weight initialization, forward propagation, backpropagation, and mini-batch gradient descent. The NumPy model is then compared against a structurally identical Keras model to validate correctness and measure the performance gap.

---

## Architecture

```
Input (784) → Dense (256, ReLU) → Dense (128, ReLU) → Output (10, Softmax)
```

| Hyperparameter  | Value |
|-----------------|-------|
| Epochs          | 20    |
| Batch size      | 128   |
| Learning rate   | 0.01  |
| Optimizer       | SGD   |
| Loss function   | Categorical cross-entropy |
| Weight init     | He normal (`sqrt(2 / fan_in)`) |

---

### Components inside `neural_network.py`

| Section | Description |
|---------|-------------|
| Activation functions | `relu`, `relu_derivative`, `softmax` (numerically stable) |
| Loss function | `cross_entropy_loss` with epsilon clipping |
| `NeuralNetwork` class | `forward`, `backward`, `update_params`, `train`, `predict`, `accuracy` |
| Data loading | `load_and_preprocess_mnist` — flattens, normalizes, one-hot encodes |
| Keras benchmark | `train_keras_model` — identical architecture via `tf.keras` |
| Entry point | `main()` — trains both models and prints comparison |

---

## Requirements

```
numpy
tensorflow
```

Install with:

```bash
pip install numpy tensorflow
```

---

## Usage

```bash
python neural_network.py
```

---

## Implementation Notes

**Backpropagation**
The output layer gradient is computed as `A_last - y_true`, which is the clean analytical result when softmax and cross-entropy are combined. Hidden layer gradients are propagated using the chain rule with the ReLU derivative.

**Numerical stability**
- Softmax subtracts the row-wise maximum before exponentiation to prevent overflow.
- Cross-entropy clips predictions to `[1e-12, 1 - 1e-12]` to avoid `log(0)`.

**Weight initialization**
He initialization (`scale = sqrt(2 / fan_in)`) is used throughout to prevent vanishing/exploding gradients with ReLU activations.

---

## What to Expect

The NumPy model reaches roughly **97% test accuracy**, closely matching the Keras baseline. Training time is significantly slower since NumPy runs on CPU without graph optimization or hardware acceleration — this gap is expected and intentional; the goal is understanding, not speed.