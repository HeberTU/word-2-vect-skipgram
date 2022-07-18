# -*- coding: utf-8 -*-
"""This module includes the skipgram implementations.

Created on: 18/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import torch
from torch import nn

from word2vect.ml.networks.features import Features
from word2vect.ml.networks.fully_connected import (
    HiddenLayers,
    OutputLayer,
    build_sequential_layers,
)


class SkipGram(nn.Module):
    """Skipgram architecture for Word2Vect task."""

    def __init__(
        self,
        features: Features,
        hidden_layers: HiddenLayers,
        output_layer: OutputLayer,
    ):
        """Create ans instance of SkipGram class.

        Args:
            features: Input features definition.
            hidden_layers: Hidden leyers definition
            output_layer: Output layer definition.
        """
        super().__init__()
        self.features = features
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

        self.embeddings = nn.Embedding(
            num_embeddings=self.features.vocabulary.size,
            embedding_dim=self.features.embedding_dim,
        )

        self.fc_sequential = build_sequential_layers(
            self.features, self.hidden_layers, self.output_layer
        )

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass implementation.

        Args:
            x: feature batch.

        Returns:
            output: network predictions.
        """
        x = self.embeddings(x)
        output = self.fc_sequential(x)
        return output
