"""Visualization utilities for optimizer study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .optimizers import OptimizationTrace


def plot_optimization_traces(
    traces: dict[str, OptimizationTrace],
    title: str = "Optimization Convergence",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot loss curves for multiple optimizers."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, trace in traces.items():
        ax.plot(trace.loss, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_2d_contour(
    f,
    traces: dict[str, OptimizationTrace],
    xlim: tuple[float, float] = (-5, 5),
    ylim: tuple[float, float] = (-5, 5),
    title: str = "Optimizer Trajectories",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """2D contour plot with optimizer trajectories overlaid."""
    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([xi, yi]))[0] for xi, yi in zip(xr, yr)] for xr, yr in zip(X, Y)])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(X, Y, Z, levels=30, cmap="viridis", alpha=0.6)

    colors = plt.cm.tab10(np.linspace(0, 1, len(traces)))
    for (name, trace), color in zip(traces.items(), colors):
        pts = np.array(trace.params)
        ax.plot(pts[:, 0], pts[:, 1], "o-", markersize=2, label=name, color=color, linewidth=1.5)
        ax.plot(pts[0, 0], pts[0, 1], "s", color=color, markersize=8)  # start
        ax.plot(pts[-1, 0], pts[-1, 1], "*", color=color, markersize=12)  # end

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_grad_norms(
    traces: dict[str, OptimizationTrace],
    title: str = "Gradient Norms",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot gradient norm over time for each optimizer."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, trace in traces.items():
        ax.plot(trace.grad_norms, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("‖∇f‖")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(
    histories: dict[str, dict[str, list[float]]],
    title: str = "Neural Network Training",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot loss and accuracy curves for multiple optimizer runs on a neural net."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories.items():
        axes[0].plot(hist["loss"], label=name)
        if any(a > 0 for a in hist["accuracy"]):
            axes[1].plot(hist["accuracy"], label=name)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].legend()

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_loss_landscape(
    net,
    X: np.ndarray,
    y: np.ndarray,
    direction1: list[np.ndarray],
    direction2: list[np.ndarray],
    n_points: int = 25,
    alpha_range: tuple[float, float] = (-1.0, 1.0),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """3D surface plot of the loss landscape around current parameters.

    Parameters
    ----------
    net : NeuralNetwork
        The trained network.
    direction1, direction2 : list[np.ndarray]
        Random directions in parameter space (same shape as net.get_params()).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    base_params = net.get_params()
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    betas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    Z = np.zeros((n_points, n_points))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            perturbed = [p + a * d1 + b * d2 for p, d1, d2 in zip(base_params, direction1, direction2)]
            net.set_params(perturbed)
            output, _ = net.forward(X)
            Z[i, j] = net.compute_loss(output, y)

    net.set_params(base_params)  # restore

    A, B = np.meshgrid(alphas, betas)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, Z.T, cmap="viridis", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_zlabel("Loss")
    ax.set_title("Loss Landscape")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_initialization_comparison(
    results: dict[str, dict[str, list[float]]],
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Compare training curves for different weight initialization strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, hist in results.items():
        axes[0].plot(hist["loss"], label=name)
        if any(a > 0 for a in hist["accuracy"]):
            axes[1].plot(hist["accuracy"], label=name)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss by Initialization")
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy by Initialization")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
