from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float
    x2: float


class Output(BaseModel):
    y: float


class NeuronType(StrEnum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class LayerType(StrEnum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"
