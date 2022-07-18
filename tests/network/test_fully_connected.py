# -*- coding: utf-8 -*-
"""This module test the fully connected builder.

Created on: 18/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Tuple

import pytest

from word2vect.ml.networks.features import Features
from word2vect.ml.networks.fully_connected import (
    HiddenLayers,
    OutputLayer,
    build_sequential_layers,
)


@pytest.mark.parametrize(
    "network_definition",
    [
        {
            "vocabulary_size": 20,
            "embedding_dim": 10,
            "hidden_dim": [5],
            "dropout": 0.2,
        }
    ],
    indirect=True,
)
def test_build_sequential_layers(
    network_definition: Tuple[Features, HiddenLayers, OutputLayer]
) -> None:
    """Test that the sequential model has the right size and properties."""
    features, hidden_layers, output_layer = network_definition

    sequential_model = build_sequential_layers(
        features, hidden_layers, output_layer
    )

    assert len(sequential_model) == (len(hidden_layers.hidden_dim) + 1) * 3
    assert hidden_layers.activation in sequential_model
    assert output_layer.activation in sequential_model
