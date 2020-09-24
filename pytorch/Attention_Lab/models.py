import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.rnn = nn.RNN(dim_input, dim_hidden, 1, nonlinearity='relu')
        self.W = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        h_all, h_final = self.rnn(x)
        return self.W(h_final.squeeze(0))


class GRU(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.gru = nn.GRU(dim_input, dim_hidden, 1)
        self.W = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        h_all, h_final = self.gru(x)
        return self.W(h_final.squeeze(0))


class LSTM(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.lstm = nn.LSTM(dim_input, dim_hidden, 1)
        self.W = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        h_all, (h_final, c_final) = self.lstm(x)
        return self.W(h_final.squeeze(0))


class Attention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_emb):
        super().__init__()
        self.Q = nn.Linear(dim_q, dim_emb)
        self.K = nn.Linear(dim_k, dim_emb)
        self.V = nn.Linear(dim_k, dim_emb)

    def forward(self, query, key, value):
        query = self.Q(query)
        key = self.K(key)
        value = self.V(value)

        a = (query.unsqueeze(1) * key.unsqueeze(0)).sum(-1)
        a = torch.softmax(a, dim=1)

        output = (a.unsqueeze(-1) * value.unsqueeze(0)).sum(1)
        return output


class AttentionGRU(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.gru = nn.GRU(dim_input, dim_hidden, 1)
        self.W = nn.Linear(dim_hidden, dim_output)
        # self.attention = nn.MultiheadAttention(dim_hidden, num_heads=1)
        self.attention = Attention(dim_hidden, dim_hidden * 2, dim_hidden)
        self.position_embeddings = nn.Embedding(512, dim_hidden)

    def forward(self, x):
        h_all, h_final = self.gru(x)
        pos = self.position_embeddings.weight[:len(x)].unsqueeze(1).expand(-1, x.shape[1], -1)
        key = torch.cat((x, pos), -1)
        query = h_final  # .view(h_final.shape[1], 1, -1)
        # attn_output, attn_output_weights = self.attention(query, key, key)
        attn_output = self.attention(query, key, key)
        return (attn_output.squeeze(0))
