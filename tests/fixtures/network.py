# -*- coding: utf-8 -*-
"""Networks fixtures.

Created on: 4/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import string
from typing import (
    Any,
    Dict,
)

from torch.nn import (
    LogSoftmax,
    ReLU,
)

from word2vect.ml import networks


def get_network_artifacts(
    network_architecture: networks.NetworkArchitecture,
) -> Dict[str, Any]:
    """Create network artifacts to initialize it.

    Args:
        network_architecture: Network architecture type.

    Returns:
        network_artifacts: network artifacts to initialize it.
    """
    vocabulary_size = 20
    vocabulary_to_idx = {
        char: idx
        for idx, char in enumerate(string.ascii_letters[:vocabulary_size])
    }
    idx_to_vocabulary = {idx: char for char, idx in vocabulary_to_idx.items()}

    _implementations = {
        networks.NetworkArchitecture.SKIPGRAM: {
            "vocabulary_size": vocabulary_size,
            "embedding_dim": 10,
            "hidden_dim": [5],
            "dropout": 0.2,
            "vocabulary_to_idx": vocabulary_to_idx,
            "idx_to_vocabulary": idx_to_vocabulary,
            "activation": ReLU(),
            "activation_out": LogSoftmax(dim=1),
        }
    }

    network_artifacts = _implementations.get(network_architecture, None)

    return network_artifacts


def get_network_config(
    network_architecture: networks.NetworkArchitecture,
    network_artifacts: Dict[str, Any],
) -> networks.NetworkConfig:
    """Get network config using the provided network artifacts.

    Args:
        network_architecture: network architecture.
        network_artifacts: network artifacts.

    Returns:
        network_config: network configuration.
    """
    network_config = networks.NetworkConfig(
        network_architecture=network_architecture,
        features=networks.Features(
            vocabulary=networks.Vocabulary(
                size=network_artifacts.get("vocabulary_size"),
                vocabulary_to_idx=network_artifacts.get("vocabulary_to_idx"),
                idx_to_vocabulary=network_artifacts.get("idx_to_vocabulary"),
            ),
            embedding_dim=network_artifacts.get("embedding_dim"),
        ),
        hidden_layers=networks.HiddenLayers(
            hidden_dim=network_artifacts.get("hidden_dim"),
            activation=network_artifacts.get("activation"),
            dropout=network_artifacts.get("dropout"),
        ),
        output_layer=networks.OutputLayer(
            activation=network_artifacts.get("activation_out")
        ),
    )
    return network_config
