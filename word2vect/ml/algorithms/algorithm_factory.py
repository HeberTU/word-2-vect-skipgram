# -*- coding: utf-8 -*-
"""This module contains the factory for algorithms.

Created on: 19/9/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import enum
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Type,
)

import torch
from torch.utils.data import DataLoader

from word2vect.ml import models
from word2vect.ml.algorithms.algorithm import Algorithm
from word2vect.ml.algorithms.network_algorithm import NetworkAlgorithm


class AlgorithmType(enum.Enum):
    """Available algorithm types."""

    INTERFACE = enum.auto()
    NETWORK = enum.auto()


@dataclass(frozen=True)
class AlgorithmConfig:
    """Data structure to store loss function config."""

    algorithm_type: AlgorithmType
    nn_model: Type[models.NNModel]
    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None


class AlgorithmFactory:
    """Algorithm Factory class.

    This is a creational class used to instantiate the different algorithm
    implementations.
    """

    def __init__(self, algorithm_config: AlgorithmConfig) -> None:
        """Instantiate the algorithm factory.

        Args:
            algorithm_config: algorithm configuration.
        """
        self._config = algorithm_config
        self._algorithms = {AlgorithmType.NETWORK: NetworkAlgorithm}

    def create(self) -> Algorithm:
        """Create the algorithm instance.

        Returns:
            algorithm: algorithm instance.
        """
        algorithm = self._algorithms.get(self._config.algorithm_type, None)

        if algorithm is None:
            raise NotImplementedError(
                f"{self._config.algorithm_type} not implemented."
            )

        return algorithm(
            self.config.nn_model,
            self.config.train_loader,
            self.config.val_loader,
            self.config.scheduler,
        )

    @property
    def config(self) -> AlgorithmConfig:
        """Get factory config."""
        return self._config
