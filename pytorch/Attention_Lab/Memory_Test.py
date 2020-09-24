import matplotlib
import torch.optim
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from models import RNN, GRU, LSTM, AttentionGRU
from synthetic_data import RandomSequenceDataset

input_dimension = 4
hidden_dimension = 4
sequence_lengths = [1, 3, 5, 10, 20, 40, 100, 200, 400]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(model, optimizer, sequence_length, dimension=input_dimension, batch_size=512):
    dataset = RandomSequenceDataset(dimension, sequence_length, batch_size)
    max_loss = 10
    num_epochs = 100
    epoch_size = 5
    best_loss = 1e10
    for epoch in range(num_epochs):
        average_loss = 0
        for i in range(epoch_size):
            x = dataset.get_batch().to(dev)
            y = x[0]
            y_h = model(x)
            loss = F.mse_loss(y_h, y)
            if np.isnan(loss.item()):
                average_loss += max_loss / epoch_size
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss += loss.clamp(0, max_loss).item() / epoch_size
        if average_loss < best_loss:
            best_loss = average_loss
    return best_loss


def plot_losses(losses, sequence_lengths):
    plt.plot(losses)
    plt.xlabel('Sequence Length')
    plt.xticks(range(len(sequence_lengths)), labels=sequence_lengths)
    plt.ylabel('MSE')
    plt.show()


opt = lambda model: torch.optim.Adam(model.parameters(), lr=0.1)


def memory_test(model_func, sequence_lengths):
    losses = []
    for sequence_length in sequence_lengths:
        model = model_func().to(dev)
        optimizer = opt(model)
        loss = train_model(model, optimizer, sequence_length=sequence_length)
        print(f'Sequence length: {sequence_length}, \t running average loss: {loss}')
        losses.append(loss)
    plot_losses(losses, sequence_lengths)


memory_test(lambda: RNN(input_dimension, hidden_dimension, input_dimension), sequence_lengths)
memory_test(lambda: GRU(input_dimension, hidden_dimension, input_dimension), sequence_lengths)
memory_test(lambda: LSTM(input_dimension, hidden_dimension, input_dimension), sequence_lengths)
memory_test(lambda: AttentionGRU(input_dimension, hidden_dimension, input_dimension), sequence_lengths)

