"""Tests for neural network implementation."""

import numpy as np
import pytest

from src.models import NeuralNetwork, train_network, relu, relu_derivative, sigmoid, sigmoid_derivative
from src.optimizers import Adam, SGD


class TestActivations:
    def test_relu_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_relu_negative(self):
        x = np.array([-1.0, -2.0, 0.0])
        np.testing.assert_array_equal(relu(x), [0.0, 0.0, 0.0])

    def test_relu_derivative(self):
        x = np.array([-1.0, 0.5, 2.0])
        np.testing.assert_array_equal(relu_derivative(x), [0.0, 1.0, 1.0])

    def test_sigmoid_range(self):
        x = np.array([-10.0, 0.0, 10.0])
        s = sigmoid(x)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_sigmoid_at_zero(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


class TestNeuralNetwork:
    def test_forward_shape(self):
        net = NeuralNetwork([784, 128, 10], seed=42)
        X = np.random.randn(32, 784)
        output, cache = net.forward(X)
        assert output.shape == (32, 10)

    def test_output_sums_to_one(self):
        net = NeuralNetwork([784, 128, 10], task='classification', seed=42)
        X = np.random.randn(5, 784)
        output, _ = net.forward(X)
        np.testing.assert_allclose(output.sum(axis=1), np.ones(5), atol=1e-6)

    def test_predict_shape(self):
        net = NeuralNetwork([784, 128, 10], seed=42)
        X = np.random.randn(10, 784)
        preds = net.predict(X)
        assert preds.shape == (10,)
        assert all(0 <= p <= 9 for p in preds)

    def test_backward_shapes(self):
        net = NeuralNetwork([4, 3, 2], seed=42)
        X = np.random.randn(8, 4)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        output, cache = net.forward(X)
        w_grads, b_grads = net.backward(output, y, cache)
        assert w_grads[0].shape == net.weights[0].shape
        assert w_grads[1].shape == net.weights[1].shape
        assert b_grads[0].shape == net.biases[0].shape

    def test_get_set_params_roundtrip(self):
        net = NeuralNetwork([4, 3, 2], seed=42)
        params = net.get_params()
        net2 = NeuralNetwork([4, 3, 2], seed=99)
        net2.set_params(params)
        for w1, w2 in zip(net.weights, net2.weights):
            np.testing.assert_array_equal(w1, w2)

    def test_regression_task(self):
        net = NeuralNetwork([4, 8, 1], task='regression', seed=42)
        X = np.random.randn(16, 4)
        output, _ = net.forward(X)
        assert output.shape == (16, 1)

    def test_loss_decreases_with_training(self):
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)
        net = NeuralNetwork([4, 8, 2], task='classification', seed=42)
        hist = train_network(
            net, X, y,
            lambda n: [Adam(lr=0.01) for _ in range(n)],
            epochs=20, batch_size=32,
        )
        assert hist['loss'][-1] < hist['loss'][0]
