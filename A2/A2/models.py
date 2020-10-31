import torch
import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        """
        :param vocab_size: the number of different embeddings to make (need one embedding for every unique word).
        :param embedding_dim: the dimension of each embedding vector.
        :param num_classes: the number of target classes.
        :param word_embeddings: optional pre-trained word embeddings. If not given word embeddings are trained from
        random initialization. If given then provided word_embeddings are used and the embeddings are not trained.
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        self.W = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        :param x: a LongTensor of shape [batch_size, max_sequence_length]. Each row is one sequence (movie review),
        the i'th element in a row is the (integer) ID of the i'th token in the original text.
        :return: a FloatTensor of shape [batch_size, num_classes]. Predicted class probabilities for every sequence
        in the batch.
        """

        # TODO perform embed, aggregate, and linear, then return the predicted class probabilities.
        word_embeds = self.embeddings(x)
        sum = torch.sum(word_embeds, 1)
        max = torch.max(word_embeds, 1)[0]
        count = (word_embeds != 0).sum(dim=1)
        mean = torch.div(sum, count)
        # print(torch.sum(word_embeds, 1))
        y_h = self.W(max)
        return y_h


class MultiLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        self.W1 = nn.Linear(embedding_dim, embedding_dim)
        self.W2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        word_embeds = self.embeddings(x)
        max = torch.max(word_embeds, 1)[0]
        y_hidden = self.W1(max)
        y_output = self.W2(y_hidden)
        return y_output


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, embedding_dim, 1)
        self.W = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        word_embeds = self.embeddings(x)
        # max = torch.max(word_embeds, 1)[0]
        h_all, h_final = self.gru(word_embeds)
        return self.W(h_final.squeeze(0))


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, 1)
        self.W = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        word_embeds = self.embeddings(x)
        max = torch.max(word_embeds, 1)[0]
        h_all, (h_final, c_final) = self.lstm(max)
        return self.W(h_final.squeeze(0))
