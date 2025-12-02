from typing import List

from nn.nn import XORNeuralNetwork
from nn.constant import INPUT, OUTPUT
from nn.schemas import History
from tqdm import tqdm


def train(
    neural_network: XORNeuralNetwork, epochs: int = 1000, record_interval: int = 100
) -> List[History]:
    history: List[History] = []
    for epoch in tqdm(range(epochs)):
        total_loss: float = 0.0
        for input, answer in zip(INPUT, OUTPUT):
            neural_network.forward(input)
            loss: float = neural_network.backward(input=input, target=answer)
            total_loss += loss
        if epoch % record_interval == 0:
            print(f"Epoch {epoch} - Loss: {total_loss / len(INPUT)}")
            history.append(
                History(
                    epoch=epoch,
                    loss=total_loss / len(INPUT),
                    input_to_hidden_weights=neural_network.W1.copy(),
                    hidden_to_output_weights=neural_network.W2.copy(),
                    input_to_hidden_bias=neural_network.bias1.copy(),
                    hidden_to_output_bias=neural_network.bias2.copy(),
                )
            )
    return history
