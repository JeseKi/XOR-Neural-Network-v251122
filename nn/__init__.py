from .nn import XORNeuralNetwork
from .schemas import Input, Target, History
from .constant import INPUT, OUTPUT
from .train import train
from .viz import plot_loss, animate_weights
from .activation import ActivationType
from .loss import LossType

__all__ = [
    "XORNeuralNetwork",
    "Input",
    "Target",
    "INPUT",
    "OUTPUT",
    "train",
    "History",
    "plot_loss",
    "animate_weights",
    "ActivationType",
    "LossType",
]
