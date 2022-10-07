# -*- coding: utf-8 -*-
"""Test module for algorithm.

Created on: 19/9/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Type

import pytest

from word2vect.ml import (
    algorithms,
    models,
    networks,
    tracker,
)

TEST_PARAMS = [
    {
        "batch_size": 128,
        "dataset_size": 2000,
        "algorithm_type": algorithms.AlgorithmType.NETWORK,
        "model_type": models.ModelType.WORD2VECT,
        "network_architecture": networks.NetworkArchitecture.SKIPGRAM,
    },
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "training_tracker",
    [tracker.TrackerType.TENSORBOARD],
    indirect=True,
)
@pytest.mark.parametrize("algorithm", TEST_PARAMS, indirect=True)
def test_network_algorithm_model_runs(
    training_tracker: tracker.TrainingTracker,
    algorithm: Type[algorithms.Algorithm],
):
    """Test if the algorithm run by inspecting the performed steps."""
    algorithm.run(training_tracker)
    assert algorithm.nn_model.step == 7
