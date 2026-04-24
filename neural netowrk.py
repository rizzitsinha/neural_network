import time
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  1. ACTIVATION FUNCTIONS & THEIR DERIVATIVES
# ═══════════════════════════════════════════════════════════════════════════

def relu(z):
    """Rectified Linear Unit: max(0, z)"""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 where z > 0, else 0."""
    return (z > 0).astype(np.float64)


def softmax(z):
    """
    Numerically-stable softmax.
    Subtract the row-wise max to prevent overflow in exp().
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
#  2. LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def cross_entropy_loss(y_pred, y_true):

    # Clip predictions to avoid log(0)
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]
    return loss


# ═══════════════════════════════════════════════════════════════════════════
#  3. NEURAL NETWORK CLASS
# ═══════════════════════════════════════════════════════════════════════════

class NeuralNetwork:

    def __init__(self, layer_dims):
        self.num_layers = len(layer_dims) - 1  
        self.weights = []
        self.biases = []

        np.random.seed(42)
        for i in range(self.num_layers):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            scale = np.sqrt(2.0 / fan_in)
            w = np.random.randn(fan_in, fan_out) * scale
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    # ─── Forward Pass ──────────────────────────────────────────────────
    def forward(self, X):

        cache = {"Z": [], "A": [X]}
        A = X

        for i in range(self.num_layers):
            Z = A @ self.weights[i] + self.biases[i]   # linear transform
            cache["Z"].append(Z)

            if i < self.num_layers - 1:
                # Hidden layer → ReLU
                A = relu(Z)
            else:
                # Output layer → Softmax
                A = softmax(Z)

            cache["A"].append(A)

        return A, cache

    # ─── Backward Pass (Backpropagation) ───────────────────────────────
    def backward(self, y_true, cache):

        batch_size = y_true.shape[0]
        grads_w = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        # ── Output Layer Gradient ──────────────────────────────────────

        dZ = cache["A"][-1] - y_true  # shape: (batch, num_classes)

        for i in reversed(range(self.num_layers)):
            A_prev = cache["A"][i]  # activation from the previous layer

            # Gradient w.r.t. weights
            grads_w[i] = (A_prev.T @ dZ) / batch_size

            # Gradient w.r.t. biases
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / batch_size

            if i > 0:
                # ── Propagate error to the previous layer ──────────────

                dA = dZ @ self.weights[i].T
                dZ = dA * relu_derivative(cache["Z"][i - 1])

        return grads_w, grads_b

    # ─── Parameter Update (Gradient Descent) ───────────────────────────
    def update_params(self, grads_w, grads_b, learning_rate):
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

    # ─── Training Loop ─────────────────────────────────────────────────
    def train(self, X_train, y_train, X_val, y_val,
              epochs=20, batch_size=128, learning_rate=0.01):

        num_samples = X_train.shape[0]
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            num_batches = 0

            # ── Process Mini-Batches ───────────────────────────────────
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # 1) Forward pass
                y_pred, cache = self.forward(X_batch)

                # 2) Compute loss
                loss = cross_entropy_loss(y_pred, y_batch)
                epoch_loss += loss
                num_batches += 1

                # 3) Backward pass
                grads_w, grads_b = self.backward(y_batch, cache)

                # 4) Update parameters
                self.update_params(grads_w, grads_b, learning_rate)

            # ── Epoch-level Metrics ────────────────────────────────────
            avg_train_loss = epoch_loss / num_batches
            val_pred, _ = self.forward(X_val)
            val_loss = cross_entropy_loss(val_pred, y_val)
            val_acc = self.accuracy(X_val, y_val)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"  Epoch {epoch:3d}/{epochs}  │  "
                  f"Train Loss: {avg_train_loss:.4f}  │  "
                  f"Val Loss: {val_loss:.4f}  │  "
                  f"Val Acc: {val_acc:.4f}")

        return history

    # ─── Prediction & Accuracy ─────────────────────────────────────────
    def predict(self, X):
        """Return predicted class indices."""
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y_onehot):
        """Compute classification accuracy."""
        preds = self.predict(X)
        true_labels = np.argmax(y_onehot, axis=1)
        return np.mean(preds == true_labels)


# ═══════════════════════════════════════════════════════════════════════════
#  4. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def load_and_preprocess_mnist():

    from tensorflow.keras.datasets import mnist

    (X_train, y_train_raw), (X_test, y_test_raw) = mnist.load_data()

    # Flatten 28×28 images into 784-dim vectors
    X_train = X_train.reshape(-1, 784).astype(np.float64)
    X_test = X_test.reshape(-1, 784).astype(np.float64)

    # Normalise pixel values to [0, 1]
    X_train /= 255.0
    X_test /= 255.0

    # One-hot encode labels (10 classes)
    num_classes = 10
    y_train_oh = np.eye(num_classes)[y_train_raw]
    y_test_oh = np.eye(num_classes)[y_test_raw]

    return X_train, y_train_oh, X_test, y_test_oh, y_train_raw, y_test_raw


# ═══════════════════════════════════════════════════════════════════════════
#  5. TENSORFLOW / KERAS BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

def train_keras_model(X_train, y_train, X_test, y_test,
                      epochs=20, batch_size=128):

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Silence verbose TF logging
    tf.get_logger().setLevel("ERROR")

    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(256, activation="relu",
                     kernel_initializer="he_normal"),
        layers.Dense(128, activation="relu",
                     kernel_initializer="he_normal"),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n  Training Keras model …")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_acc


# ═══════════════════════════════════════════════════════════════════════════
#  6. MAIN — TRAIN, EVALUATE & COMPARE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  MNIST Neural Network — NumPy vs TensorFlow/Keras Benchmark")
    print("=" * 72)

    # ── Hyperparameters ────────────────────────────────────────────────
    LAYER_DIMS    = [784, 256, 128, 10]
    EPOCHS        = 20
    BATCH_SIZE    = 128
    LEARNING_RATE = 0.01

    print(f"\n  Architecture : {' → '.join(str(d) for d in LAYER_DIMS)}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")

    # ── Load Data ──────────────────────────────────────────────────────
    print("\n  Loading MNIST dataset …")
    (X_train, y_train_oh, X_test, y_test_oh,
     y_train_raw, y_test_raw) = load_and_preprocess_mnist()
    print(f"  Train samples: {X_train.shape[0]:,}")
    print(f"  Test  samples: {X_test.shape[0]:,}")

    # ── Train NumPy Model ──────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  Training NumPy Neural Network")
    print("─" * 72)

    nn = NeuralNetwork(LAYER_DIMS)
    start = time.time()
    history = nn.train(
        X_train, y_train_oh,
        X_test, y_test_oh,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )
    numpy_time = time.time() - start
    numpy_acc = nn.accuracy(X_test, y_test_oh)

    print(f"\n  ✓ NumPy model — Test Accuracy : {numpy_acc:.4f}")
    print(f"                   Training Time : {numpy_time:.1f}s")

    # ── Train Keras Model ──────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  Training TensorFlow/Keras Benchmark")
    print("─" * 72)

    start = time.time()
    keras_acc = train_keras_model(
        X_train, y_train_oh,
        X_test, y_test_oh,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    keras_time = time.time() - start

    print(f"\n  ✓ Keras model — Test Accuracy : {keras_acc:.4f}")
    print(f"                  Training Time : {keras_time:.1f}s")

if __name__ == "__main__":
    main()

