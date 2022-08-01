# -*- coding: utf-8 -*-
"""Training tracker factory module.

The factory module is a creational design pattern that provides an interface
for creating Training trackers.

Created on: 1/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import enum

from word2vect.ml import metrics

from .tensorboard_tracker import TensorboardTracker
from .tracker_interface import TrainingTracker


class TrackerType(enum.Enum):
    """Available trackers."""

    TENSORBOARD = enum.auto()


class TrainingTrackerFactory:
    """Training tracker factory to instantiate training tracker objects."""

    def __init__(self):
        """Instantiate tracker factory."""
        self.training_trackers = {TrackerType.TENSORBOARD: TensorboardTracker}

    def create(
        self,
        tracker_type: TrackerType,
        models_repr: metrics.ModelsRepr,
        steps_per_epoch: int,
    ) -> TrainingTracker:
        """Create the training tracker.

        Args:
            tracker_type: training tracker type.
            models_repr: Data structure that stores the model representation.
            steps_per_epoch: The total number of steps_per_epoch.

        Returns:
            tracker: training tracker.
        """
        tracker = self.training_trackers.get(tracker_type, None)

        if tracker is None:
            raise NotImplementedError(f"{tracker_type} not implemented-")

        return tracker(models_repr, steps_per_epoch)
