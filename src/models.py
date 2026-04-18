"""From-scratch neural network with backpropagation.

A minimal feedforward network built entirely in NumPy to demonstrate
how optimizers interact with neural network training.
"""

import numpy as np

from .optimizers import softmax


# ── Activation functions ──────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


# ── Neural Network ───────────────────────────────────────────────────────

class NeuralNetwork:
    """Feedforward neural network with configurable hidden layers.

    Supports ReLU hidden activations and softmax output for classification,
    or linear output for regression.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of each layer including input and output. E.g. [784, 128, 10].
    activation : str
        Hidden layer activation: 'relu' or 'sigmoid'.
    task : str
        'classification' (softmax + cross-entropy) or 'regression' (linear + MSE).
    seed : int | None
        Random seed for weight initialization.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        task: str = "classification",
        seed: int | None = None,
    ):
        self.layer_sizes = layer_sizes
        self.task = task
        self.n_layers = len(layer_sizes) - 1

        if activation == "relu":
            self.act_fn = relu
            self.act_deriv = relu_derivative
        elif activation == "sigmoid":
            self.act_fn = sigmoid
            self.act_deriv = sigmoid_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")

        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He initialization for ReLU, Xavier for sigmoid
            if activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0, scale, (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Forward pass. Returns (output, cache_list) for backprop."""
        cache = []
        a = X

        for i in range(self.n_layers - 1):
            z = a @ self.weights[i] + self.biases[i]
            cache.append({"a_prev": a, "z": z})
            a = self.act_fn(z)

        # Output layer (no hidden activation)
        z = a @ self.weights[-1] + self.biases[-1]
        cache.append({"a_prev": a, "z": z})

        if self.task == "classification":
            output = softmax(z)
        else:
            output = z

        return output, cache

    def compute_loss(self, output: np.ndarray, y: np.ndarray) -> float:
        """Compute loss given forward output and targets."""
        if self.task == "classification":
            # y is integer labels
            log_probs = -np.log(output[np.arange(len(y)), y] + 1e-12)
            return float(np.mean(log_probs))
        else:
            return float(np.mean((output - y) ** 2))

    def backward(self, output: np.ndarray, y: np.ndarray, cache: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Backpropagation. Returns (weight_grads, bias_grads)."""
        m = output.shape[0]
        w_grads = [None] * self.n_layers
        b_grads = [None] * self.n_layers

        # Output layer gradient
        if self.task == "classification":
            # Softmax + cross-entropy: dL/dz = probs - one_hot
            dz = output.copy()
            dz[np.arange(m), y] -= 1
            dz /= m
        else:
            dz = 2 * (output - y) / m

        for i in reversed(range(self.n_layers)):
            a_prev = cache[i]["a_prev"]
            w_grads[i] = a_prev.T @ dz
            b_grads[i] = np.sum(dz, axis=0)

            if i > 0:
                da = dz @ self.weights[i].T
                dz = da * self.act_deriv(cache[i - 1]["z"])

        return w_grads, b_grads

    def get_params(self) -> list[np.ndarray]:
        """Flatten all parameters into a list."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w)
            params.append(b)
        return params

    def set_params(self, params: list[np.ndarray]) -> None:
        """Set parameters from a flat list."""
        idx = 0
        for i in range(self.n_layers):
            self.weights[i] = params[idx]
            self.biases[i] = params[idx + 1]
            idx += 2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass returning predictions (class labels or values)."""
        output, _ = self.forward(X)
        if self.task == "classification":
            return np.argmax(output, axis=1)
        return output

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        preds = self.predict(X)
        return float(np.mean(preds == y))


def train_network(
    net: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    optimizer_factory,
    epochs: int = 50,
    batch_size: int = 64,
) -> dict[str, list[float]]:
    """Train network with given optimizer. Returns per-epoch metrics.

    Parameters
    ----------
    net : NeuralNetwork
        Network to train (modified in-place).
    optimizer_factory : callable
        Function returning a list of optimizer instances (one per parameter).
        E.g. lambda: [Adam(lr=0.001) for _ in range(n_params)]
    """
    n_params = 2 * net.n_layers
    optimizers = optimizer_factory(n_params)
    history = {"loss": [], "accuracy": []}
    n = len(X_train)

    for _ in range(epochs):
        # Shuffle
        perm = np.random.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            X_b = X_shuf[start:start + batch_size]
            y_b = y_shuf[start:start + batch_size]

            output, cache = net.forward(X_b)
            loss = net.compute_loss(output, y_b)
            w_grads, b_grads = net.backward(output, y_b, cache)

            # Apply optimizer to each parameter
            params = net.get_params()
            grads = []
            for wg, bg in zip(w_grads, b_grads):
                grads.append(wg)
                grads.append(bg)

            new_params = []
            for p, g, opt in zip(params, grads, optimizers):
                new_params.append(opt.step(p, g))
            net.set_params(new_params)

            epoch_loss += loss
            n_batches += 1

        history["loss"].append(epoch_loss / n_batches)
        if net.task == "classification":
            history["accuracy"].append(net.accuracy(X_train, y_train))
        else:
            history["accuracy"].append(0.0)

    return history
