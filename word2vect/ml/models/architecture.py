# -*- coding: utf-8 -*-
"""This module will be deprecated.
Created on Mon Aug 10 16:42:58 2020

@author: HTRUJILLO
"""

from torch import nn


class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()

        self.embed = nn.Embedding(
            num_embeddings=n_vocab, embedding_dim=n_embed
        )
        self.fc = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        scores = self.fc(x)
        log_ps = self.log_softmax(scores)

        return log_ps
