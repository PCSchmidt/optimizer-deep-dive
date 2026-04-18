# Demo Guide: Optimizer Deep Dive

## What This Project Demonstrates

A pure-NumPy implementation of three gradient-based optimizers (BGD, SGD+Momentum, Adam), a feedforward neural network with backpropagation, and systematic comparisons on analytical functions and MNIST.

## Running the Demo

### 1. Setup (~1 min)
```bash
pip install -e ".[dev]"
```

### 2. Run the Notebook (~5 min on CPU)
```bash
jupyter notebook notebooks/optimizer_deep_dive.ipynb
```
Execute all cells. MNIST downloads automatically via scikit-learn (~15 MB).

### 3. Key Outputs to Inspect

| Section | What to Look For |
|---------|-----------------|
| §3-5 Basics | Each optimizer converging on simple quadratics |
| §6 Showdown | 2D trajectory plots on Rosenbrock — Adam navigates the valley efficiently |
| §6 Condition Study | Table showing convergence degradation as κ increases |
| §8 MNIST NN | Adam reaching ~93% vs SGD ~85% on handwritten digits |
| §9 Initialization | He init vs large random — wrong init prevents learning |
| §10 Loss Landscape | 3D surface showing the curvature around trained solution |

### 4. Run Tests
```bash
python -m pytest tests/ -v
```

## Talking Points

1. **Pure NumPy — no autograd.** Every gradient is computed analytically, including full backpropagation through a multi-layer network. This demonstrates deep understanding of the chain rule in practice.

2. **The condition number story.** Convergence rate of GD is directly tied to κ = λ_max/λ_min. Adam's adaptive rates effectively reduce the effective condition number.

3. **Initialization is not optional.** He initialization for ReLU networks, Xavier for sigmoid. Wrong initialization causes vanishing gradients and complete training failure.

4. **Math ↔ Code bridge.** Every algorithm is presented first as a mathematical equation, then as NumPy code, then evaluated empirically.

## Evidence Files

After running the notebook, `evidence/` contains:
- `rosenbrock_trajectories.png` — 2D optimizer paths on Rosenbrock
- `optimizer_showdown.png` — Convergence comparison
- `gradient_norms.png` — Gradient magnitude over time
- `nn_training_comparison.png` — MNIST training curves
- `initialization_sensitivity.png` — He vs Xavier vs large random
- `loss_landscape.png` — 3D loss surface
