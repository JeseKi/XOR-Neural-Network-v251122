from typing import Callable

import numpy as np
from numpy.random._generator import Generator
from numpy import random

from nn.schemas import Input, Target


class XORNeuralNetwork:
    def __init__(self, seed: int = 42, learning_rate: float = 0.01):
        # init constant
        self.rng: Generator = random.default_rng(seed)
        self.learning_rate: float = learning_rate

        self.sigmoid: Callable[[np.ndarray], np.ndarray] = lambda x: 1 / (1 + np.exp(-x))
        self.sigmoid_derivative: Callable[[np.ndarray], np.ndarray] = lambda x: x * (1 - x)
        self.mean_squared_error: Callable[[float, float], float] = (
            lambda output, answer: 1 / 2 * (output - answer) ** 2
        )
        self.mean_squared_error_derivative: Callable[[float, float], float] = (
            lambda output, answer: output - answer
        )

        # init variable
        self.W1: np.ndarray = self.rng.uniform(size=(2, 2), low=-1, high=1)
        self.W1_pred: np.ndarray = np.zeros(shape=(1,2))
        self.W1_bias: np.ndarray = self.rng.uniform(size=(1,2), low=-1, high=1)

        self.W2: np.ndarray = self.rng.uniform(size=(2, 1), low=-1, high=1)
        self.W2_pred: np.ndarray = np.zeros(shape=(1,1))
        self.W2_bias: np.ndarray = self.rng.uniform(size=(1,1), low=-1, high=1)

    def forward(self, input: Input) -> Target:
        input_arr = np.array([[input.x1, input.x2]]) # shape(1, 2)

        z1 = input_arr @ self.W1 + self.W1_bias # shape(1, 2)
        h1 = self.sigmoid(z1)
        self.W1_pred = h1

        z2 = h1 @ self.W2 + self.W2_bias # shape(1, 1)
        h2 = self.sigmoid(z2)
        self.W2_pred = h2

        return Target(y=h2[0])

    def backward(self, input: Input, target: Target) -> float:
        input_hidden_arr = np.array([[input.x1, input.x2]]) # shape(1, 2)
        loss = self.mean_squared_error(self.W2_pred[0], target.y)

        grad_loss = self.mean_squared_error_derivative(
            self.W2_pred[0], target.y
        )
        grad_output_outputs = self.sigmoid_derivative(
            self.W2_pred[0]
        )
        grad_output = grad_loss * grad_output_outputs # shape(1, 1)
        grad_output_weights = grad_output * self.W1_pred # shape(1, 2)
        grad_output_bias = grad_output # shape(1, 1)

        grad_hidden_outputs = self.sigmoid_derivative(self.W1_pred) # shape(1, 2)
        grad_hidden = (grad_output @ self.W2.T) * grad_hidden_outputs # shape(1, 2)
        grad_hidden_weights = input_hidden_arr.T @ grad_hidden # shape(2, 2)
        grad_hidden_bias = grad_hidden # shape(1, 2)

        self.W1 -= self.learning_rate * grad_hidden_weights
        self.W1_bias -= self.learning_rate * grad_hidden_bias
        self.W2 -= self.learning_rate * grad_output_weights.T
        self.W2_bias -= self.learning_rate * grad_output_bias

        return loss
