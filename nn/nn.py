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
        
        # init variable
        self.input_to_hidden_weights: np.ndarray = self.rng.uniform(size=(2, 2))
        self.input_to_hidden_outputs: np.ndarray = np.zeros(shape=(2,))
        self.input_to_hiden_bias: np.ndarray = self.rng.uniform(size=(2,))
        
        self.hidden_to_output_weights: np.ndarray = self.rng.uniform(size=(1, 2))
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
        
        z5 = h1 * self.hidden_to_output_weights[0][0] + self.hidden_to_output_bias[0]
        z6 = h2 * self.hidden_to_output_weights[0][1] + self.hidden_to_output_bias[0]
        
        h3 = self.sigmoid(z5 + z6)
        self.hidden_to_output_outputs = np.array([h3])
        
        return Output(y = h3)