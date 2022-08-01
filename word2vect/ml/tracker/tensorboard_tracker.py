# -*- coding: utf-8 -*-
"""Tensorboard tracker implementation.

Created on: 1/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import os
import pathlib
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from word2vect.ml import (
    loss_functions,
    metrics,
)
from word2vect.ml.tracker.tracker_interface import (
    Stage,
    TrainingTracker,
)


def _log_model_repr(
    model_repr: metrics.ModelsRepr, directory: pathlib.Path
) -> None:
    """Save model representation in a human-readable format.

    Args:
        model_repr: Model representation.
        directory: directory where the model repr will be saved.

    Returns:
        None
    """
    os.makedirs(directory, exist_ok=True)
    if (directory / "model_repr.txt").is_file():
        return None

    with open(directory / "model_repr.txt", "w") as file:
        file.write(model_repr.raw_repr)


def _add_version(base_dir: pathlib.Path) -> pathlib.Path:
    """Add timestamp to base directory."""
    version = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return base_dir / version


class TensorboardTracker(TrainingTracker):
    """Tensorboard tracker."""

    def __init__(
        self,
        models_repr: metrics.ModelsRepr,
        steps_per_epoch: int,
    ):
        """Instantiate a TensorboardTracker object.

        Args:
            models_repr: Model representation, it helps to properly save model
            metrics in the right directory.
            steps_per_epoch: Number of steps per epoch.
        """
        self.is_network_graph_logged = False
        self._steps_per_epoch = steps_per_epoch

        # Iteration helper variables.
        self._epoch_count: int = 0
        self._base_steps: int = 0
        self._base_epochs: int = 0
        self._stage: Stage = Stage.TRAIN

        base_dir = (
            self.get_base_dir() / "tracker_logs" / models_repr.hashed_repr
        )
        self._log_dir = _add_version(base_dir)
        _log_model_repr(models_repr, base_dir)
        self._writer = SummaryWriter(log_dir=self._log_dir)

        self._model_metrics: List[metrics.ModelMetrics] = []

    @staticmethod
    def get_base_dir() -> pathlib.Path:
        """Get project's base directory."""
        return pathlib.Path(__file__).parents[3]

    @property
    def base_steps(self):
        """Get base steps."""
        return self._base_steps

    @base_steps.setter
    def base_steps(self, base_steps: int) -> None:
        """Set base steps."""
        self._base_steps = base_steps

    @property
    def base_epochs(self):
        """Get base epochs."""
        return self._base_epochs

    @base_epochs.setter
    def base_epochs(self, base_epochs: int) -> None:
        """Set base epochs."""
        self._base_epochs = base_epochs

    @property
    def total_epoch_count(self) -> int:
        """Get the total epochs count."""
        return self.base_steps + self.epoch_count * self._steps_per_epoch

    @property
    def epoch_count(self) -> int:
        """Get the epochs count."""
        return self._epoch_count

    @epoch_count.setter
    def epoch_count(self, epoch_count: int) -> None:
        """Set epochs count."""
        self._epoch_count = epoch_count

    def reset_epoch(self) -> None:
        """Reset epoch count to zero."""
        self.epoch_count = 0

    @property
    def steps_per_epoch(self) -> int:
        """Get steps per epoch."""
        return self._steps_per_epoch

    @property
    def stage(self) -> Stage:
        """Get tracking stage."""
        return self._stage

    @stage.setter
    def stage(self, stage: Stage) -> None:
        """Set tracking stage."""
        self._stage = stage

    def flush(self) -> None:
        """Flush writer metrics."""
        self._writer.flush()

    def log_metrics(
        self, model_metrics: metrics.ModelMetrics, model_name: str, step: int
    ) -> None:
        """Log model metrics values.

        Args:
            model_metrics: model metrics data structure.
            model_name: model name.
            step: step that triggers the tracking.

        Returns:
            None.
        """
        current_step = self.total_epoch_count + step
        self._model_metrics.append(model_metrics)
        step = (
            current_step
            if self.stage == Stage.TRAIN
            else self.base_epochs + self.epoch_count
        )

        for metric in model_metrics.get_metrics():
            tag = f"{self.stage.name}/{model_name}/{metric.name}"
            tag_hist = f"{self.stage.name}/{model_name}/hist/{metric.name}"
            metric_value = metric.average_value
            self._writer.add_scalar(tag, metric_value, step)
            self._writer.add_histogram(
                tag_hist, np.array(metric.metric_values.values), step
            )

    def log_training_stats(
        self,
        training_stats: loss_functions.TrainingStats,
        model_name: str,
        step: int,
    ) -> None:
        """Log the training stats stores in the TrainingStats data structure.

        Args:
            training_stats: Training stats.
            model_name: model name.
            step: step that triggers the tracking.

        Returns:
            None
        """
        current_step = self.total_epoch_count + step
        step = (
            current_step
            if self.stage == Stage.TRAIN
            else self.base_epochs + self.epoch_count
        )
        for name, stat in training_stats.get_training_stats():
            tag = f"{self.stage.name}/{model_name}/{name}"
            tag_hist = f"{self.stage.name}/{model_name}/hist/{name}"
            avg_stat = torch.cat(stat).float().mean().item()

            self._writer.add_scalar(tag, avg_stat, step)
            self._writer.add_histogram(
                tag_hist, torch.cat(stat).detach().numpy(), step
            )

    def log_weights(
        self, model: nn.Module, model_name: str, step: int
    ) -> None:
        """Log model weights.

        Args:
            model: ml model.
            model_name: model name.
            step: step that triggers the tracking.

        Returns:
            None.
        """
        current_step = self.total_epoch_count + step
        for name, params in model.named_parameters():
            tag = f"weights/{model_name}/{name}"
            self._writer.add_histogram(tag, params, current_step)

    def log_grads(self, model: nn.Module, model_name: str, step: int) -> None:
        """Log model gradients.

        Args:
            model: model
            model_name: model name
            step: step that triggers the tracking.

        Returns:
            None
        """
        current_step = self.total_epoch_count + step
        for name, param in model.named_parameters():
            tag = f"grads/{model_name}/{name}"
            self._writer.add_histogram(tag, param.grad, current_step)

    def log_network(self, model: nn.Module, batch: torch.Tensor) -> None:
        """Log network architecture.

        Args:
            model: model
            batch: model name.

        Returns:
            None.
        """
        self._writer.add_graph(model, batch)
        self.is_network_graph_logged = True

    def log_learning_rate(
        self, optimizer: torch.optim.Optimizer, model_name: str, step: int
    ) -> None:
        """Log model's learning rate.

        Args:
            optimizer: model optimizer.
            model_name: model name
            step: step that triggers the tracking.

        Returns:
            None
        """
        current_step = self.total_epoch_count + step
        lr = optimizer.param_groups[0]["lr"]
        tag = f"{self.stage.name}/LR/{model_name}"
        self._writer.add_scalar(tag, lr, current_step)
