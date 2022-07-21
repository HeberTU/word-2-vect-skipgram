# -*- coding: utf-8 -*-
"""This module test the interface loss function data structures.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Tuple

import pytest
import torch

import word2vect.ml.loss_functions as loss_functions


@pytest.mark.unit
def test_training_stats_updates(
    loss_artifacts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    training_stats: loss_functions.TrainingStats,
) -> None:
    """Test TrainingStats update method."""
    loss, prediction, target = loss_artifacts

    assert len(training_stats.loss) == 0
    assert len(training_stats.prediction) == 0
    assert len(training_stats.target) == 0

    training_stats.update(loss=loss, prediction=prediction, target=target)

    assert len(training_stats.loss) > 0
    assert len(training_stats.prediction) > 0
    assert len(training_stats.target) > 0


@pytest.mark.unit
def test_training_stats_flush(
    loss_artifacts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    training_stats: loss_functions.TrainingStats,
) -> None:
    """Test TrainingStats flush method."""
    loss, prediction, target = loss_artifacts

    training_stats.update(loss=loss, prediction=prediction, target=target)

    assert len(training_stats.loss) > 0
    assert len(training_stats.prediction) > 0
    assert len(training_stats.target) > 0

    training_stats.flush()

    assert len(training_stats.loss) == 0
    assert len(training_stats.prediction) == 0
    assert len(training_stats.target) == 0
