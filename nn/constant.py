from typing import List

from nn.schemas import Input, Output

INPUT: List[Input] = [
    Input(x1=0, x2=0),
    Input(x1=0, x2=1),
    Input(x1=1, x2=0),
    Input(x1=1, x2=1),
]  # XOR Input

OUTPUT: List[Output] = [
    Output(y=0),
    Output(y=1),
    Output(y=1),
    Output(y=0),
]  # XOR Answer
