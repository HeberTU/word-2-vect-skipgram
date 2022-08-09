# -*- coding: utf-8 -*-
"""Model factory module.

Created on: 2/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import optim

from word2vect.ml import (
    loss_functions,
    metrics,
    networks,
)

from .model_interface import (
    ModelConfig,
    ModelType,
)
from .word2vect_model import Word2VectModel


@dataclass
class OptimizerConfig:
    """Optimization configuration definitions."""

    learning_rate: float
    weight_decay: Optional[float] = None
    momentum: Optional[float] = None


@dataclass
class ModelDefinition:
    """Data structure to store model definition."""

    model_config: ModelConfig
    network_config: networks.NetworkConfig
    metrics_config: metrics.MetricsConfig
    loss_function_config: loss_functions.LossFunctionConfig
    optimizer_config: OptimizerConfig


class ModelFactory:
    """Model factory.

    Currently it only supports word2vect model.
    """

    def __init__(self, model_definition: ModelDefinition) -> ModelFactory:
        """Instantiate an ModelFactory object.

        Args:
            model_definition: model definition data structure.
        """
        self._model_definition = model_definition
        self.models = {ModelType.WORD2VECT: Word2VectModel}

    def create(self) -> Word2VectModel:
        """Create a Model instance.

        Returns:
            model: model.
        """
        network = networks.NetworkFactory(
            self._model_definition.network_config
        ).create()

        optimizer = optim.SGD(
            params=network.parameters(),
            lr=self._model_definition.optimizer_config.learning_rate,
            momentum=self._model_definition.optimizer_config.momentum,
            weight_decay=self._model_definition.optimizer_config.weight_decay,
        )

        model_metrics = metrics.ModelMetricsFactory(
            metrics_config=self._model_definition.metrics_config
        ).create()

        loss_function = loss_functions.LossFunctionFactory(
            loss_function_config=self._model_definition.loss_function_config
        ).create()

        model_cls = self.models.get(
            self._model_definition.model_config.model_type, None
        )
        if model_cls is None:
            raise NotImplementedError(
                f"{self._model_definition.model_config.model_type} not "
                "implemented"
            )

        return model_cls(
            network=network,
            model_metrics=model_metrics,
            model_config=self._model_definition.model_config,
            optimizer=optimizer,
            loss_function=loss_function,
        )
