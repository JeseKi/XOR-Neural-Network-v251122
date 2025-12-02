from typing import Callable

import numpy as np
from numpy.random._generator import Generator
from numpy import random

from nn.schemas import Input, Output


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
        self.hidden_weights: np.ndarray = self.rng.uniform(size=(2, 2))
        self.hidden_outputs: np.ndarray = np.zeros(shape=(1,2))
        self.hidden_bias: np.ndarray = self.rng.uniform(size=(1,2))

        self.output_weights: np.ndarray = self.rng.uniform(size=(2, 1))
        self.output_outputs: np.ndarray = np.zeros(shape=(1,1))
        self.output_bias: np.ndarray = self.rng.uniform(size=(1,1))

    def forward(self, input: Input) -> Output:
        input_arr = np.array([[input.x1, input.x2]]) # shape(1, 2)

        z1 = input_arr @ self.hidden_weights + self.hidden_bias # shape(1, 2)
        h1 = self.sigmoid(z1)
        self.hidden_outputs = h1

        z2 = h1 @ self.output_weights + self.output_bias # shape(1, 1)
        h2 = self.sigmoid(z2)
        self.output_outputs = h2

        return Output(y=h2[0])

    def backward(self, input: Input, answer: Output) -> float:
        input_hidden_arr = np.array([[input.x1, input.x2]]) # shape(1, 2)
        loss = self.mean_squared_error(self.output_outputs[0], answer.y)

        delta_loss = self.mean_squared_error_derivative(
            self.output_outputs[0], answer.y
        )
        delta_output_outputs = self.sigmoid_derivative(
            self.output_outputs[0]
        )
        delta_output = delta_loss * delta_output_outputs # shape(1, 1)
        delta_output_weights = delta_output * self.hidden_outputs # shape(1, 2)
        delta_b3 = delta_output # shape(1, 1)

        delta_hidden_outputs = self.sigmoid_derivative(self.hidden_outputs) # shape(1, 2)
        delta_hidden = (delta_output @ self.output_weights.T) * delta_hidden_outputs # shape(1, 2)
        delta_hidden_weights = input_hidden_arr.T @ delta_hidden # shape(2, 2)
        delta_hidden_bias = delta_hidden # shape(1, 2)

        self.hidden_weights -= self.learning_rate * delta_hidden_weights
        self.hidden_bias -= self.learning_rate * delta_hidden_bias
        self.output_weights -= self.learning_rate * delta_output_weights.T
        self.output_bias -= self.learning_rate * delta_b3

        return loss
