import torch


class RandomSequenceDataset():
    def __init__(self, dimension, sequence_length, batch_size, variance=1.0):
        self.dim = dimension
        self.sl = sequence_length
        self.bs = batch_size
        self.var = variance

    def get_batch(self):
        return torch.randn((self.sl, self.bs, self.dim)) * self.var
