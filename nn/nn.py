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

        self.sigmoid: Callable[[float], float] = lambda x: 1 / (1 + np.exp(-x))
        self.sigmoid_derivative: Callable[[float], float] = lambda x: x * (1 - x)
        self.mean_squared_error: Callable[[float, float], float] = (
            lambda output, answer: 1 / 2 * (output - answer) ** 2
        )
        self.mean_squared_error_derivative: Callable[[float, float], float] = (
            lambda output, answer: output - answer
        )

        # init variable
        self.input_to_hidden_weights: np.ndarray = self.rng.uniform(size=(2, 2))
        self.input_to_hidden_outputs: np.ndarray = np.zeros(shape=(2,))
        self.input_to_hiden_bias: np.ndarray = self.rng.uniform(size=(2,))

        self.hidden_to_output_weights: np.ndarray = self.rng.uniform(size=(2,))
        self.hidden_to_output_outputs: np.ndarray = np.zeros(shape=(1,))
        self.hidden_to_output_bias: np.ndarray = self.rng.uniform(size=(1,))

    def forward(self, input: Input) -> Output:
        z1 = input.x1 * self.input_to_hidden_weights[0][0] + self.input_to_hiden_bias[0]
        z2 = input.x2 * self.input_to_hidden_weights[0][1] + self.input_to_hiden_bias[0]
        z3 = input.x1 * self.input_to_hidden_weights[1][0] + self.input_to_hiden_bias[1]
        z4 = input.x2 * self.input_to_hidden_weights[1][1] + self.input_to_hiden_bias[1]

        h1 = self.sigmoid(z1 + z2)
        h2 = self.sigmoid(z3 + z4)
        self.input_to_hidden_outputs = np.array([h1, h2])

        z5 = h1 * self.hidden_to_output_weights[0] + self.hidden_to_output_bias[0]
        z6 = h2 * self.hidden_to_output_weights[1] + self.hidden_to_output_bias[0]

        h3 = self.sigmoid(z5 + z6)
        self.hidden_to_output_outputs = np.array([h3])

        return Output(y=h3)

    def backward(self, input: Input, answer: Output) -> float:
        loss = self.mean_squared_error(self.hidden_to_output_outputs[0], answer.y)

        gradient_loss = self.mean_squared_error_derivative(
            self.hidden_to_output_outputs[0], answer.y
        )
        gradient_output_local = self.sigmoid_derivative(
            self.hidden_to_output_outputs[0]
        )
        gradient_output = gradient_loss * gradient_output_local
        gradient_output_w1 = gradient_output * self.input_to_hidden_outputs[0]
        gradient_output_w2 = gradient_output * self.input_to_hidden_outputs[1]
        gradient_b3 = gradient_output

        gradient_n1_local = self.sigmoid_derivative(self.input_to_hidden_outputs[0])
        gradient_n2_local = self.sigmoid_derivative(self.input_to_hidden_outputs[1])

        gradient_n1 = (
            gradient_output * self.hidden_to_output_weights[0] * gradient_n1_local
        )
        gradient_n2 = (
            gradient_output * self.hidden_to_output_weights[1] * gradient_n2_local
        )

        gradient_n1_w1 = gradient_n1 * input.x1
        gradient_n1_w2 = gradient_n1 * input.x2
        gradient_b1 = gradient_n1 * 1

        gradient_n2_w1 = gradient_n2 * input.x1
        gradient_n2_w2 = gradient_n2 * input.x2
        gradient_b2 = gradient_n2 * 1

        self.hidden_to_output_weights[0] -= self.learning_rate * gradient_output_w1
        self.hidden_to_output_weights[1] -= self.learning_rate * gradient_output_w2
        self.hidden_to_output_bias[0] -= self.learning_rate * gradient_b3

        self.input_to_hidden_weights[0][0] -= self.learning_rate * gradient_n1_w1
        self.input_to_hidden_weights[0][1] -= self.learning_rate * gradient_n1_w2
        self.input_to_hiden_bias[0] -= self.learning_rate * gradient_b1

        self.input_to_hidden_weights[1][0] -= self.learning_rate * gradient_n2_w1
        self.input_to_hidden_weights[1][1] -= self.learning_rate * gradient_n2_w2
        self.input_to_hiden_bias[1] -= self.learning_rate * gradient_b2

        return loss
