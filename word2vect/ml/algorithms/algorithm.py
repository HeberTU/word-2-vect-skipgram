# -*- coding: utf-8 -*-
"""Algorithm interface.

Created on: 17/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from abc import (
    ABC,
    abstractmethod,
)

import torch

from word2vect.ml import (
    loss_functions,
    metrics,
    tracker,
)


class Algorithm(ABC):
    """Algorithm abstract class."""

    @abstractmethod
    def run(self, training_tracker: tracker.TrainingTracker) -> None:
        """Run training algorithm.

        Args:
            training_tracker: training tracker.

        Returns:
            None.
        """
        raise NotImplementedError()

    @abstractmethod
    def validate(self) -> None:
        """Validate model using validation data set.

        Forward step without computing the gradients and then use the
        validation set to measure the different metrics.

        Returns:
            None.
        """
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
    def log(
        self, training_tracker: tracker.TrainingTracker, stage: tracker.Stage
    ) -> None:
        """Log metrics using a tracker.

        Args:
            training_tracker: training tracker.
            stage: training stage.

        Returns:
            None.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def models_repr(self) -> metrics.ModelsRepr:
        """Get the model representation.

        Returns:
            model_repr: model representation.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def total_steps(self) -> int:
        """Get total number of steps performed by the algorithm in an epoch.

        Returns:
            total_steps: total steps performed by the model.
        """
        raise NotImplementedError()
