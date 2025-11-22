from .nn import XORNeuralNetwork
from .schemas import Input, Output, History
from .constant import INPUT, OUTPUT
from .train import train
from .viz import plot_loss, animate_weights

__all__ = [
    "XORNeuralNetwork",
    "Input",
    "Output",
    "INPUT",
    "OUTPUT",
    "train",
    "History",
    "plot_loss",
    "animate_weights",
]
