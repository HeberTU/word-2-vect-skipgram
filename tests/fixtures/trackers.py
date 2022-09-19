# -*- coding: utf-8 -*-
"""Training tracker fixtures.

Created on: 19/9/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Type

from word2vect.ml import (
    metrics,
    tracker,
)


def get_tensorboard_tracker() -> tracker.TrainingTracker:
    """Instantiate a tensorboard tracker.

    The model representation here is just a placeholder.
    """
    return tracker.TrainingTrackerFactory().get_tracker(
        "tensorboard",
        metrics.ModelsRepr("Model", "hash"),
        steps_per_epoch=10,
        tracker_log_dir="test_tmp",
    )


def get_training_tracker(tracker_type: str) -> Type[tracker.TrainingTracker]:
    """Get the training tracker instance according to the tracker type.

    Parameters
    ----------
    tracker_type: str
        training tracker type.

    Returns
    -------
    _tracker: Type[TrainingTracker]
        training tracker for the given tracker_type
    """
    tracker_dict = {"tensorboard": get_tensorboard_tracker()}

    _tracker = tracker_dict.get(tracker_type)
    if _tracker is None:
        raise NotImplementedError(
            f"Model test type for {tracker_type} not implemented"
        )

    return _tracker
