"""Tests for optimizer implementations."""

import numpy as np
import pytest

from src.optimizers import (
    Adam,
    BatchGD,
    SGD,
    OptimizationTrace,
    minimize,
    mse_loss,
    mse_grad,
    softmax,
    cross_entropy_loss,
)


class TestBatchGD:
    def test_step_reduces_params(self):
        opt = BatchGD(lr=0.1)
        params = np.array([2.0, 3.0])
        grad = np.array([1.0, 1.0])
        new = opt.step(params, grad)
        np.testing.assert_allclose(new, [1.9, 2.9])

    def test_zero_gradient_no_change(self):
        opt = BatchGD(lr=0.5)
        params = np.array([1.0, 2.0])
        grad = np.zeros(2)
        new = opt.step(params, grad)
        np.testing.assert_allclose(new, params)

    def test_converges_on_quadratic(self):
        def quadratic(x):
            return float(np.sum(x**2)), 2 * x
        trace = minimize(quadratic, np.array([5.0, 5.0]), BatchGD(lr=0.1), n_steps=100)
        assert trace.loss[-1] < 1e-6


class TestSGD:
    def test_no_momentum_matches_bgd(self):
        sgd = SGD(lr=0.1, momentum=0.0)
        bgd = BatchGD(lr=0.1)
        params = np.array([3.0, 4.0])
        grad = np.array([1.0, 2.0])
        np.testing.assert_allclose(sgd.step(params, grad), bgd.step(params, grad))

    def test_momentum_accelerates(self):
        sgd_no = SGD(lr=0.01, momentum=0.0)
        sgd_mom = SGD(lr=0.01, momentum=0.9)
        def quadratic(x):
            return float(np.sum(x**2)), 2 * x
        x0 = np.array([5.0, 5.0])
        trace_no = minimize(quadratic, x0, sgd_no, n_steps=100)
        trace_mom = minimize(quadratic, x0, sgd_mom, n_steps=100)
        assert trace_mom.loss[-1] < trace_no.loss[-1]

    def test_reset_clears_velocity(self):
        sgd = SGD(lr=0.1, momentum=0.9)
        params = np.array([1.0])
        sgd.step(params, np.array([1.0]))
        assert sgd._velocity is not None
        sgd.reset()
        assert sgd._velocity is None


class TestAdam:
    def test_first_step(self):
        adam = Adam(lr=0.1)
        params = np.array([5.0])
        grad = np.array([2.0])
        new = adam.step(params, grad)
        # First step should move params toward 0
        assert new[0] < params[0]

    def test_converges_on_quadratic(self):
        def quadratic(x):
            return float(np.sum(x**2)), 2 * x
        trace = minimize(quadratic, np.array([5.0, 5.0]), Adam(lr=0.1), n_steps=200)
        assert trace.loss[-1] < 1e-6

    def test_reset_clears_state(self):
        adam = Adam(lr=0.1)
        adam.step(np.array([1.0]), np.array([1.0]))
        assert adam._t == 1
        adam.reset()
        assert adam._t == 0
        assert adam._m is None

    def test_bias_correction(self):
        adam = Adam(lr=0.1, beta1=0.9, beta2=0.999)
        params = np.array([5.0])
        grad = np.array([2.0])
        new = adam.step(params, grad)
        # With bias correction, first step should be approximately lr
        assert abs(params[0] - new[0] - 0.1) < 0.01


class TestLossFunctions:
    def test_mse_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse_loss(y, y) == pytest.approx(0.0)

    def test_mse_known_value(self):
        pred = np.array([1.0, 2.0])
        true = np.array([0.0, 0.0])
        assert mse_loss(pred, true) == pytest.approx(2.5)

    def test_mse_grad_shape(self):
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([0.0, 0.0, 0.0])
        grad = mse_grad(pred, true)
        assert grad.shape == pred.shape

    def test_softmax_sums_to_one(self):
        logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        probs = softmax(logits)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_softmax_numerical_stability(self):
        logits = np.array([[1000.0, 1001.0, 1002.0]])
        probs = softmax(logits)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)


class TestMinimize:
    def test_trace_length(self):
        def f(x):
            return float(x[0]**2), 2 * x
        trace = minimize(f, np.array([3.0]), BatchGD(lr=0.1), n_steps=50)
        assert len(trace.loss) == 50
        assert len(trace.params) == 50
        assert len(trace.grad_norms) == 50

    def test_loss_decreases(self):
        def f(x):
            return float(np.sum(x**2)), 2 * x
        trace = minimize(f, np.array([5.0, 5.0]), BatchGD(lr=0.1), n_steps=50)
        assert trace.loss[-1] < trace.loss[0]
