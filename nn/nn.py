
import numpy as np
from numpy.random._generator import Generator
from numpy import random

from nn.schemas import Input, Target
from nn.activation import ActivationType, activation, derivative
from nn.loss import LossType, loss, grad


class XORNeuralNetwork:
    def __init__(self, seed: int = 42, learning_rate: float = 0.01):
        # init constant
        self.rng: Generator = random.default_rng(seed)
        self.learning_rate: float = learning_rate

        self.activation_type: ActivationType = ActivationType.LEAKY_RELU
        self.loss_type: LossType = LossType.MSE

        # init variable
        self.W1: np.ndarray = self.rng.uniform(size=(2, 2), low=-1, high=1)
        self.h1: np.ndarray = np.zeros(shape=(1,2))
        self.bias1: np.ndarray = self.rng.uniform(size=(1,2), low=-1, high=1)

        self.W2: np.ndarray = self.rng.uniform(size=(2, 1), low=-1, high=1)
        self.h2: np.ndarray = np.zeros(shape=(1,1))
        self.bias2: np.ndarray = self.rng.uniform(size=(1,1), low=-1, high=1)

    def forward(self, input: Input) -> Target:
        input_arr = np.array([[input.x1, input.x2]]) # shape(1, 2)

        z1 = input_arr @ self.W1 + self.bias1 # shape(1, 2)
        h1 = activation(self.activation_type, z1)
        self.h1 = h1

        z2 = h1 @ self.W2 + self.bias2 # shape(1, 1)
        h2 = activation(self.activation_type, z2)
        self.h2 = h2

        return Target(y=h2[0])

    def backward(self, input: Input, target: Target) -> float:
        input_hidden_arr = np.array([[input.x1, input.x2]]) # shape(1, 2)
        _loss = loss(self.loss_type, self.h2[0], np.array([target.y]))

        grad_loss = grad(self.loss_type, self.h2[0], np.array([target.y]))
        grad_h2 = derivative(self.activation_type, self.h2[0])
        grad_output = grad_loss * grad_h2 # shape(1, 1)
        grad_W2 = grad_output * self.h1 # shape(1, 2)
        grad_bias2 = grad_output # shape(1, 1)

        grad_h1 = derivative(self.activation_type, self.h1) # shape(1, 2)
        grad_hidden = (grad_output @ self.W2.T) * grad_h1 # shape(1, 2)
        grad_W1 = input_hidden_arr.T @ grad_hidden # shape(2, 2)
        grad_bias1 = grad_hidden # shape(1, 2)

        self.W1 -= self.learning_rate * grad_W1
        self.bias1 -= self.learning_rate * grad_bias1
        self.W2 -= self.learning_rate * grad_W2.T
        self.bias2 -= self.learning_rate * grad_bias2

        return _loss

class NeuralNetwork:
    def __init__(self, seed: int = 42, learning_rate: float = 0.01) -> None:
        self.rng: Generator = random.default_rng(seed)
        self.learn_rate: float = learning_rate