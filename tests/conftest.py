# -*- coding: utf-8 -*-
"""Testing fixtures.

Created on: 18/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from string import ascii_letters
from typing import Tuple

import pytest
from _pytest.fixtures import FixtureRequest
from torch.nn import (
    LogSoftmax,
    ReLU,
)

import tests.fixtures as w2v_fixtures
import word2vect.ml.loss_functions as loss_functions
from word2vect.ml.networks.features import (
    Features,
    Vocabulary,
)
from word2vect.ml.networks.fully_connected import (
    HiddenLayers,
    OutputLayer,
)
from word2vect.ml.networks.skipgram import SkipGram


def get_features(vocabulary_size: int, embedding_dim: int) -> Features:
    """Get feature sample test."""
    vocabulary = Vocabulary(
        size=vocabulary_size,
        vocabulary_to_idx={
            ch: idx for idx, ch in enumerate(ascii_letters[:vocabulary_size])
        },
        idx_to_vocabulary={
            idx: ch for idx, ch in enumerate(ascii_letters[:vocabulary_size])
        },
    )

    return Features(vocabulary=vocabulary, embedding_dim=embedding_dim)


def get_hidden_layers(hidden_dim: int, dropout: float) -> HiddenLayers:
    """Get hidden layer sample test."""
    return HiddenLayers(
        hidden_dim=hidden_dim, activation=ReLU(), dropout=dropout
    )


def get_output_layer():
    """Get output layer sample test."""
    return OutputLayer(activation=LogSoftmax(dim=1))


@pytest.fixture
def network_definition(
    request: FixtureRequest,
) -> Tuple[Features, HiddenLayers, OutputLayer]:
    """Generate a network definition."""
    hidden_dim = request.param.get("hidden_dim")
    dropout = request.param.get("dropout")
    vocabulary_size = request.param.get("vocabulary_size")
    embedding_dim = request.param.get("embedding_dim")

    features = get_features(vocabulary_size, embedding_dim)

    hidden_layers = get_hidden_layers(hidden_dim, dropout)

    output_layer = get_output_layer()

    return features, hidden_layers, output_layer


@pytest.fixture
def skipgram(request: FixtureRequest) -> SkipGram:
    """Create a skipgram model."""
    hidden_dim = request.param.get("hidden_dim")
    dropout = request.param.get("dropout")
    vocabulary_size = request.param.get("vocabulary_size")
    embedding_dim = request.param.get("embedding_dim")

    features = get_features(vocabulary_size, embedding_dim)

    hidden_layers = get_hidden_layers(hidden_dim, dropout)

    output_layer = get_output_layer()

    return SkipGram(features, hidden_layers, output_layer)


@pytest.fixture
def training_stats() -> loss_functions.TrainingStats:
    """Create a training stats instance."""
    return loss_functions.TrainingStats()


@pytest.fixture
def result(request: FixtureRequest) -> loss_functions.Result:
    """Create a result set."""
    loss_artifacts_type = request.param.get("loss_artifacts_type")
    loss_artifacts = w2v_fixtures.get_loss_artifacts(loss_artifacts_type)
    loss_artifacts = {
        k: v
        for k, v in loss_artifacts.items()
        if k in loss_functions.Result.__annotations__.keys()
    }
    return loss_functions.Result(**loss_artifacts)


@pytest.fixture
def ground_truth(request: FixtureRequest) -> loss_functions.GroundTruth:
    """Create a result set."""
    loss_artifacts_type = request.param.get("loss_artifacts_type")
    loss_artifacts = w2v_fixtures.get_loss_artifacts(loss_artifacts_type)
    loss_artifacts = {
        k: v
        for k, v in loss_artifacts.items()
        if k in loss_functions.GroundTruth.__annotations__.keys()
    }
    return loss_functions.GroundTruth(**loss_artifacts)


@pytest.fixture
def loss(request: FixtureRequest) -> loss_functions.Result:
    """Create a result set."""
    loss_artifacts_type = request.param.get("loss_artifacts_type")
    loss_artifacts = w2v_fixtures.get_loss_artifacts(loss_artifacts_type)
    return loss_artifacts.get("loss")
