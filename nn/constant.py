from typing import List

from nn.schemas import Input, Target

INPUT: List[Input] = [
    Input(x1=0, x2=0),
    Input(x1=0, x2=1),
    Input(x1=1, x2=0),
    Input(x1=1, x2=1),
]  # XOR Input

OUTPUT: List[Target] = [
    Target(y=0),
    Target(y=1),
    Target(y=1),
    Target(y=0),
]  # XOR Answer
