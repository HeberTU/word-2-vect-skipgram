# -*- coding: utf-8 -*-
"""Neural-net factory module.

The module provides the constructor factory for the different neural networks
implementations.

Created on: 18/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from dataclasses import dataclass
from enum import (
    Enum,
    auto,
)
from typing import (
    Dict,
    Type,
)

from torch import nn

from word2vect.ml.networks.features import Features
from word2vect.ml.networks.fully_connected import (
    HiddenLayers,
    OutputLayer,
)
from word2vect.ml.networks.skipgram import SkipGram


class NetworkArchitecture(Enum):
    """Net architectures avilible."""

    SKIPGRAM = auto()


@dataclass(frozen=True)
class NetworkConfig:
    """Network configuration definition."""

    network_architecture: NetworkArchitecture
    features: Features
    hidden_layers: HiddenLayers
    output_layer: OutputLayer


class NetworkFactory:
    """Network factory."""

    def __init__(self, network_config: NetworkConfig) -> None:
        """Initialize neural network factory.

        The factory is used to instantiate the net architecture based on the
        provided configurations.

        Args:
            network_config:
        """
        self._config = network_config
        self.network_dict: Dict[NetworkArchitecture, Type[nn.Module]] = {
            NetworkArchitecture.SKIPGRAM: SkipGram
        }

    def create(self) -> nn.Module:
        """Instantiate the network, given a config and net architecture.

        Returns
        -------
        nn.Module
            The network for the given model type.
        """
        network = self.network_dict.get(
            self._config.network_architecture, None
        )

        if network is None:
            msg = (
                f"{self._config.network_architecture} architecture "
                f"not implemented"
            )
            raise NotImplementedError(msg)

        return network(
            self._config.features,
            self._config.hidden_layers,
            self._config.output_layer,
        )
