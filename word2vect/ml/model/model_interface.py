# -*- coding: utf-8 -*-
"""Interface model module.

This module implement the abstract model class, which is used to define
the common interface across the different model implementations.

Created on: 1/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pathlib
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import (
    Iterable,
    Optional,
    Tuple,
)

import numpy as np
import torch

from word2vect import utils
from word2vect.ml import (
    loss_functions,
    metrics,
    tracker,
)

from .model_factory import ModelType


@dataclass()
class BatchData:
    """Data structure to store batch data."""

    word_idx: torch.Tensor
    word: Optional[np.ndarray] = None


@dataclass
class ModelConfig:
    """Data structure to store model configuration."""

    model_type: ModelType
    model_name: str
    gradient_clipping_value: Optional[float] = None
    network_weights_path: Optional[pathlib.Path] = None


class NNModel(ABC):
    """Neuran network model.

    The Neural Network model abstraction is a container where all the
    different pieces converge and interact.
    """

    @abstractmethod
    def forward(
        self,
        batch: BatchData,
        stage: tracker.Stage = tracker.Stage.SERVE,
    ) -> torch.Tensor:
        """Perform neural network pass using the given stage config."""
        raise NotImplementedError()

    @abstractmethod
    def learn(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
    ) -> torch.Tensor:
        """Update the weights of the underlying neural network based on loss.

        Args:
            result: network results produced by a forward pass.
            ground_truth: Ground truth data.

        Returns:
            loss: model loss.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
        loss: Optional[torch.Tensor] = None,
        stage: tracker.Stage = tracker.Stage.TRAIN,
    ) -> None:
        """Measure model performance using the model metrics.

        Args:
            result: network results produced by a forward pass.
            ground_truth: Ground truth data.
            loss: Loss values.
            stage: model stage.

        Returns:
            None.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_model_result(
        self,
        predictions: torch.Tensor,
        stage: tracker.Stage = tracker.Stage.TRAIN,
    ) -> loss_functions.Result:
        """Create a result instance base on net predictions.

        Args:
            predictions: predictions made by the network.
            stage: model stage.

        Returns:
            results: model results.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        """Get model optimizer."""
        raise NotImplementedError()

    @property
    def parameters(self) -> Iterable[torch.Tensor]:
        """Get model's net parameter."""
        return self.network.parameters()

    @property
    def named_parameters(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """Get model's net named parameter."""
        return self.network.named_parameters()

    @property
    def repr(self) -> metrics.ModelsRepr:
        """Get model representation."""
        return metrics.ModelsRepr(
            raw_repr=repr(self), hashed_repr=utils.hash_object_repr(self)
        )

    @property
    def step(self) -> int:
        """Get model's performed steps."""
        return self._step

    @step.setter
    def step(self, step: int) -> None:
        """Set model steps."""
        self._step = step

    @property
    def optimizing_metric_value(self):
        """Get optimizing metric average value."""
        return self.metrics.optimizing_metric.average_value

    def reset(self):
        """Reset model history."""
        self.step = 0
        self.metrics.flush()
        self.training_stats.flush()

    @abstractmethod
    def log(
        self,
        training_tracker: tracker.TrainingTracker,
        stage: tracker.Stage = tracker.Stage.TRAIN,
    ) -> None:
        """Log metrics and training stats.

        Args:
            training_tracker: training tracker.
            stage: model stage

        Returns:
            None.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_loss(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
    ) -> torch.Tensor:
        """Compute loss from model results.

        Args:
            result: network results produced by a forward pass.
            ground_truth:  Ground truth data.

        Returns:
            loss: model loss.
        """
        raise NotImplementedError()

    @abstractmethod
    def save_nn_model_parameters(self) -> None:
        """Save model torch params."""
        raise NotImplementedError()

    @abstractmethod
    def load_nn_model_parameters(self) -> None:
        """Load model parameters."""
        raise NotImplementedError()

    @abstractmethod
    def preprocess_ground_truth(
        self, ground_truth: loss_functions.GroundTruth, stage: tracker.Stage
    ) -> loss_functions.GroundTruth:
        """Preprocess ground truth.

        Args:
            ground_truth: Ground truth data.
            stage: model stage.

        Returns:
            ground_truth: preprocess ground truth.
        """
        raise NotImplementedError()

    @abstractmethod
    def preprocess_result(
        self, result: loss_functions.Result, stage: tracker.Stage
    ) -> loss_functions.Result:
        """Preprocess results.

        Args:
            result: network results produced by a forward pass.
            stage: model stage.

        Returns:
            results: preprocess results.
        """
        raise NotImplementedError()
