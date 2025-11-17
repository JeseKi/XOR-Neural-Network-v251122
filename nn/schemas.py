from typing import Optional
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    x1: float
    x2: Optional[float] = None

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