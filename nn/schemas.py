from pydantic import BaseModel, ConfigDict
import numpy as np


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float
    x2: float


class Output(BaseModel):
    y: float


class History(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_encoders={np.ndarray: lambda x: x.tolist()}
    )

    epoch: int
    loss: float
    input_to_hidden_weights: np.ndarray
    hidden_to_output_weights: np.ndarray
