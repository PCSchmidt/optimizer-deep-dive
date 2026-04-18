"""From-scratch optimizers: Batch Gradient Descent, SGD, and Adam.

All implementations use pure NumPy — no autograd frameworks.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizationTrace:
    """Records parameter history and loss at each step."""

    params: list[np.ndarray] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)


class BatchGD:
    """Batch (full) gradient descent."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return params - self.lr * grad


class SGD:
    """Stochastic gradient descent with optional momentum.

    v_t = momentum * v_{t-1} + grad
    θ_t = θ_{t-1} - lr * v_t
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity: np.ndarray | None = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self._velocity is None:
            self._velocity = np.zeros_like(params)
        self._velocity = self.momentum * self._velocity + grad
        return params - self.lr * self._velocity

    def reset(self) -> None:
        self._velocity = None


class Adam:
    """Adam optimizer (Kingma & Ba, 2015).

    m_t = β₁ m_{t-1} + (1 - β₁) g_t
    v_t = β₂ v_{t-1} + (1 - β₂) g_t²
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)
        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad ** 2
        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = self._v / (1 - self.beta2 ** self._t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self) -> None:
        self._m = None
        self._v = None
        self._t = 0


# ── Loss functions ────────────────────────────────────────────────────────

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_pred - y_true) ** 2))


def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Gradient of MSE w.r.t. y_pred."""
    return 2 * (y_pred - y_true) / y_true.size


def cross_entropy_loss(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Softmax cross-entropy loss. y_true is one-hot or integer labels."""
    probs = softmax(logits)
    if y_true.ndim == 1:
        # Integer labels
        log_probs = -np.log(probs[np.arange(len(y_true)), y_true] + 1e-12)
    else:
        log_probs = -np.sum(y_true * np.log(probs + 1e-12), axis=1)
    return float(np.mean(log_probs))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


# ── Optimization runner ──────────────────────────────────────────────────

def minimize(
    f_and_grad,
    x0: np.ndarray,
    optimizer,
    n_steps: int = 100,
) -> OptimizationTrace:
    """Run optimizer on f_and_grad(x) → (loss, grad) for n_steps.

    Parameters
    ----------
    f_and_grad : callable
        Function that takes parameters and returns (loss, gradient).
    x0 : np.ndarray
        Initial parameters.
    optimizer : BatchGD | SGD | Adam
        Optimizer instance with a .step(params, grad) method.
    n_steps : int
        Number of optimization steps.

    Returns
    -------
    OptimizationTrace
        Recorded params, loss, and gradient norms.
    """
    trace = OptimizationTrace()
    x = x0.copy()

    for _ in range(n_steps):
        loss, grad = f_and_grad(x)
        trace.params.append(x.copy())
        trace.loss.append(loss)
        trace.grad_norms.append(float(np.linalg.norm(grad)))
        x = optimizer.step(x, grad)

    return trace
