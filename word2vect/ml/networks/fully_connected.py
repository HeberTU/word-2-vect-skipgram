# -*- coding: utf-8 -*-
"""This module contains the fully connected constructor and definitions.

Created on: 18/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Union,
)

import torch.nn as nn

from word2vect.ml.networks.features import Features

Activation = Union[nn.ReLU, nn.SELU]
OutputActivation = nn.LogSoftmax


@dataclass(frozen=True)
class HiddenLayers:
    """Hidden Layer definition."""

    hidden_dim: List[int]
    activation: Activation
    dropout: Optional[float] = None


@dataclass(frozen=True)
class OutputLayer:
    """Output activation definition."""

    activation: OutputActivation


def build_sequential_layers(
    features: Features, hidden_layers: HiddenLayers, output_layer: OutputLayer
) -> nn.Sequential:
    """Create a fully connected layers set.

    Args:
        features: Input features definition.
        hidden_layers: Hidden leyers definition
        output_layer: Output layer definition.

    Returns:
        sequential: sequential model.
    """
    dropout = (
        nn.Dropout(hidden_layers.dropout)
        if hidden_layers.dropout
        else nn.Identity()
    )

    _dict = {
        "hidden": (hidden_layers.activation, dropout),
        "output": (output_layer.activation, nn.Identity()),
    }

    layers = []
    size_list: List[int] = (
        [features.embedding_dim]
        + hidden_layers.hidden_dim
        + [features.vocabulary.size]
    )

    for j in range(len(size_list) - 1):
        layer_config = "hidden" if j < len(size_list) - 2 else "output"
        act, drp = _dict.get(layer_config)
        layers += [nn.Linear(size_list[j], size_list[j + 1]), act, drp]

    return nn.Sequential(*layers)
