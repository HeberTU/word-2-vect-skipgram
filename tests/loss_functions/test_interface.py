# -*- coding: utf-8 -*-
"""This module test the interface loss function data structures.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pytest
import torch

from word2vect.ml import loss_functions

TEST_PARAMS = {
    "loss_artifacts_type": loss_functions.LossFunctionType.INTERFACE
}


@pytest.mark.unit
@pytest.mark.parametrize("result", [TEST_PARAMS], indirect=True)
@pytest.mark.parametrize("loss", [TEST_PARAMS], indirect=True)
@pytest.mark.parametrize("ground_truth", [TEST_PARAMS], indirect=True)
def test_training_stats_updates(
    training_stats: loss_functions.TrainingStats,
    result: loss_functions.Result,
    ground_truth: loss_functions.GroundTruth,
    loss: torch.Tensor,
) -> None:
    """Test TrainingStats update method."""
    assert len(training_stats.loss) == 0
    assert len(training_stats.prediction) == 0
    assert len(training_stats.target) == 0

    training_stats.update(
        loss=loss, prediction=result.prediction, target=ground_truth.target
    )

    assert len(training_stats.loss) > 0
    assert len(training_stats.prediction) > 0
    assert len(training_stats.target) > 0


@pytest.mark.unit
@pytest.mark.parametrize("result", [TEST_PARAMS], indirect=True)
@pytest.mark.parametrize("loss", [TEST_PARAMS], indirect=True)
@pytest.mark.parametrize("ground_truth", [TEST_PARAMS], indirect=True)
def test_training_stats_flush(
    training_stats: loss_functions.TrainingStats,
    result: loss_functions.Result,
    ground_truth: loss_functions.GroundTruth,
    loss: torch.Tensor,
) -> None:
    """Test TrainingStats flush method."""
    training_stats.update(
        loss=loss, prediction=result.prediction, target=ground_truth.target
    )

    assert len(training_stats.loss) > 0
    assert len(training_stats.prediction) > 0
    assert len(training_stats.target) > 0

    training_stats.flush()

    assert len(training_stats.loss) == 0
    assert len(training_stats.prediction) == 0
    assert len(training_stats.target) == 0
