# -*- coding: utf-8 -*-
"""Word-2-vect model implementation.

Created on: 3/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pathlib

from torch import nn
from torch.optim.optimizer import Optimizer

from word2vect.ml import (
    loss_functions,
    metrics,
)

from .model_interface import (
    ModelConfig,
    NNModel,
)


class Word2VectModel(NNModel):
    """Word 2 Vect Model."""

    def __init__(
        self,
        network: nn.Module,
        model_metrics: metrics.ModelMetrics,
        model_config: ModelConfig,
        optimizer: Optimizer,
        loss_function: loss_functions.LossFunction,
    ):
        """Instantiate a Word 2 vect model.

        Args:
            network: Neural network.
            model_metric: Metric that will be logged.
            model_config: Model configuration data structure.
            optimizer: Model optimizer.
            loss_function: loss function.
        """
        self.network = network
        self.model_metrics = model_metrics
        self.model_name = model_config.model_name
        self.loss_function = loss_function

        self._step: int = 0
        weights_path = model_config.network_weights_path
        path = pathlib.Path(__file__).parents[3] / "params"
        self.params_dir = weights_path if weights_path is not None else path

        self._initialize_repr_metrics()

        self.initial_repr = self.repr

        self.training_stats = loss_functions.TrainingStats()

        self._optimizer = optimizer
        self._gradient_clipping_value = model_config.gradient_clipping_value
        self.model_config = model_config

    def _initialize_repr_metrics(self):
        """Create model representation."""
        self.model_metrics._models_repr = self.repr
