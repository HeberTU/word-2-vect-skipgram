# -*- coding: utf-8 -*-
"""Interface training tracker module.

This module implement the abstract training tracker class, which is used to
define the common interface across the different training tracker
implementations.

Created on: 1/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import enum
from abc import (
    ABC,
    abstractmethod,
)

import torch
from torch import nn

from word2vect.ml import (
    loss_functions,
    metrics,
)


class Stage(enum.Enum):
    """Model stages."""

    TRAIN = enum.auto()
    SERVE = enum.auto()


class TrainingTracker(ABC):
    """Keep track of all training metrics."""

    @property
    @abstractmethod
    def base_steps(self):
        """Get base steps."""
        raise NotImplementedError()

    @base_steps.setter
    @abstractmethod
    def base_steps(self, base_steps: int) -> None:
        """Set base steps."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def base_epochs(self):
        """Get base epochs."""
        raise NotImplementedError()

    @base_epochs.setter
    @abstractmethod
    def base_epochs(self, base_epochs: int) -> None:
        """Set base epochs."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def epoch_count(self) -> int:
        """Get epoch count."""
        raise NotImplementedError()

    @abstractmethod
    def reset_epoch(self) -> None:
        """Reset epoch count."""
        raise NotImplementedError()

    @epoch_count.setter
    @abstractmethod
    def epoch_count(self, epoch_count: int) -> None:
        """Set epoch count."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def steps_per_epoch(self) -> int:
        """Get steps per epoch."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def stage(self):
        """Get model stage."""
        raise NotImplementedError()

    @stage.setter
    @abstractmethod
    def stage(self, stage: Stage) -> None:
        """Set model stage."""
        raise NotImplementedError()

    @abstractmethod
    def flush(self) -> None:
        """Flush metrics."""
        raise NotImplementedError()

    @abstractmethod
    def log_metrics(
        self, model_metrics: metrics.ModelMetrics, model_name: str, step: int
    ) -> None:
        """Log metrics."""
        raise NotImplementedError()

    @abstractmethod
    def log_training_stats(
        self,
        training_stats: loss_functions.TrainingStats,
        model_name: str,
        step: int,
    ) -> None:
        """Log learning stats."""
        raise NotImplementedError()

    @abstractmethod
    def log_weights(
        self, model: nn.Module, model_name: str, step: int
    ) -> None:
        """Log model weights."""
        raise NotImplementedError()

    @abstractmethod
    def log_grads(self, model: nn.Module, model_name: str, step: int) -> None:
        """Log model gradients."""
        raise NotImplementedError()

    @abstractmethod
    def log_network(self, model: nn.Module, batch: torch.Tensor) -> None:
        """Log model network."""
        raise NotImplementedError()

    @abstractmethod
    def log_learning_rate(
        self, optimizer: torch.optim.Optimizer, model_name: str, step: int
    ) -> None:
        """Log learning rate info."""
        raise NotImplementedError()
