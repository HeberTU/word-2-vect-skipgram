# -*- coding: utf-8 -*-
"""Network algorithm implementation.

Created on: 23/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Optional,
    Type,
)

import torch
from torch.utils.data import DataLoader

from word2vect.ml import (
    loss_functions,
    metrics,
    models,
    tracker,
)
from word2vect.ml.algorithms.algorithm import Algorithm


class NetworkAlgorithm(Algorithm):
    """The basic algorithm to train a single NN Model."""

    def __init__(
        self,
        nn_model: Type[models.NNModel],
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """Initialize a network algorithm.

        Args:
            nn_model: Neural network model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            scheduler: learning rate scheduler.
        """
        self.nn_model = nn_model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self.best_epoch_val_optimizing_metric = None
        self.scheduler = scheduler

    def run(self, training_tracker: tracker.TrainingTracker) -> None:
        """Run the algorithm from a fresh start.

        Args:
            training_tracker: Training tracker

        Returns:
            None
        """
        self.nn_model.reset()
        for batch, ground_truth in self.train_loader:
            predictions = self.nn_model.forward(
                batch, stage=tracker.Stage.TRAIN
            )
            result = self.nn_model.get_model_result(
                predictions, stage=tracker.Stage.TRAIN
            )
            loss = self.nn_model.learn(result, ground_truth)
            if self.scheduler:
                self.scheduler.step()
            self.evaluate(result, ground_truth, loss)
            self.log(training_tracker)

    def validate(self) -> None:
        """Validate model using validation data set.

        Returns:
            None.
        """
        self.nn_model.reset()
        for batch, ground_truth in self.val_loader:
            predictions = self.nn_model.forward(
                batch, stage=tracker.Stage.SERVE
            )
            result = self.nn_model.get_model_result(
                predictions, stage=tracker.Stage.SERVE
            )
            loss = self.nn_model.compute_loss(result, ground_truth)
            self.evaluate(result, ground_truth, loss, tracker.Stage.SERVE)

    def evaluate(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
        loss: torch.Tensor,
        stage: tracker.Stage = tracker.Stage.TRAIN,
    ) -> None:
        """Measure model metrics for the NN model.

        Args:
            result: model result.
            ground_truth: ground truth.
            loss: loss.
            stage: training stage.

        Returns:
            None.
        """
        self.nn_model.evaluate(result, ground_truth, loss, stage)

    def log(
        self,
        training_tracker: tracker.TrainingTracker,
        stage: tracker.Stage = tracker.Stage.TRAIN,
    ) -> None:
        """Log model running.

        Args:
            training_tracker: training tracker.
            stage: training stage.

        Returns:
            None.
        """
        self.nn_model.log(tracker, stage)

    @property
    def models_repr(self) -> metrics.ModelsRepr:
        """Get the model representation.

        Returns:
            model_repr: model representation.
        """
        return self.nn_model.initial_repr

    @property
    def total_steps(self) -> int:
        """Get total number of steps performed by the algorithm in an epoch.

        Returns:
            total_steps: total steps performed by the model.
        """
        return len(self.train_loader)

    @property
    def train_loader(self) -> DataLoader:
        """Get training loader."""
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader:
        """Get validation loader."""
        return self._val_loader

    def __repr__(self) -> str:
        """Get human-readable representation.

        Returns:
            algorithm representation.
        """
        return (
            "NetworkAlgorithm("
            f"NN model={repr(self.nn_model)}, "
            f"Training Loader={repr(self.train_loader)})"
        )
