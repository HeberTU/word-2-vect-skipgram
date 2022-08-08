# -*- coding: utf-8 -*-
"""Testing fixtures.

Created on: 18/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Dict,
    Tuple,
    Type,
)

import numpy as np
import pytest
import torch
from _pytest.fixtures import FixtureRequest
from torch import nn
from torch.nn import (
    LogSoftmax,
    ReLU,
)

import tests.fixtures as w2v_fixtures
from word2vect.ml import (
    loss_functions,
    metrics,
    model,
    networks,
)


def get_features(
    vocabulary_size: int,
    embedding_dim: int,
    vocabulary_to_idx: Dict[str, int],
    idx_to_vocabulary: Dict[int, str],
) -> networks.Features:
    """Get feature sample test."""
    vocabulary = networks.Vocabulary(
        size=vocabulary_size,
        vocabulary_to_idx=vocabulary_to_idx,
        idx_to_vocabulary=idx_to_vocabulary,
    )

    return networks.Features(
        vocabulary=vocabulary, embedding_dim=embedding_dim
    )


def get_hidden_layers(
    hidden_dim: int, dropout: float
) -> networks.HiddenLayers:
    """Get hidden layer sample test."""
    return networks.HiddenLayers(
        hidden_dim=hidden_dim, activation=ReLU(), dropout=dropout
    )


def get_output_layer() -> networks.OutputLayer:
    """Get output layer sample test."""
    return networks.OutputLayer(activation=LogSoftmax(dim=1))


@pytest.fixture
def network_definition(
    request: FixtureRequest,
) -> Tuple[networks.Features, networks.HiddenLayers, networks.OutputLayer]:
    """Generate a network definition."""
    network_architecture = request.param.get("network_architecture")

    network_artifacts = w2v_fixtures.get_network_artifacts(
        network_architecture
    )

    features = get_features(
        vocabulary_size=network_artifacts.get("vocabulary_size"),
        embedding_dim=network_artifacts.get("embedding_dim"),
        vocabulary_to_idx=network_artifacts.get("vocabulary_to_idx"),
        idx_to_vocabulary=network_artifacts.get("idx_to_vocabulary"),
    )

    hidden_layers = get_hidden_layers(
        network_artifacts.get("hidden_dim"),
        network_artifacts.get("dropout"),
    )

    output_layer = get_output_layer()

    return features, hidden_layers, output_layer


@pytest.fixture
def network(request: FixtureRequest) -> Type[nn.Module]:
    """Create a skipgram model."""
    network_architecture = request.param.get("network_architecture")

    network_artifacts = w2v_fixtures.get_network_artifacts(
        network_architecture
    )
    network_config = networks.NetworkConfig(
        features=networks.Features(
            vocabulary=networks.Vocabulary(
                size=network_artifacts.get("vocabulary_size"),
                vocabulary_to_idx=network_artifacts.get("vocabulary_to_idx"),
                idx_to_vocabulary=network_artifacts.get("idx_to_vocabulary"),
            ),
            embedding_dim=network_artifacts.get("embedding_dim"),
        ),
        hidden_layers=networks.HiddenLayers(
            hidden_dim=network_artifacts.get("hidden_dim"),
            activation=network_artifacts.get("activation"),
            dropout=network_artifacts.get("dropout"),
        ),
        output_layer=networks.OutputLayer(
            activation=network_artifacts.get("activation_out")
        ),
    )

    network = networks.NetworkFactory(network_config=network_config).create(
        network_architecture=network_architecture
    )

    return network


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


@pytest.fixture
def metric_values() -> metrics.MetricValues:
    """Create a metric values instance."""
    return metrics.MetricValues()


@pytest.fixture
def metric(request: FixtureRequest) -> Type[metrics.Metric]:
    """Create a metric values instance."""
    metric_type = request.param.get("metric_type")

    metrics_artifacts = w2v_fixtures.get_metrics_artifacts(
        metric_type=metric_type,
    )

    _metric = metrics.MetricFactory(
        metric_config=metrics.MetricConfig(
            metric_type=metric_type,
            params=metrics_artifacts.get("params"),
        )
    ).create()

    return _metric


@pytest.fixture
def measurement(request: FixtureRequest) -> metrics.Measurement:
    """Create a measurement set."""
    metric_type = request.param.get("metric_type")
    metrics_artifacts = w2v_fixtures.get_metrics_artifacts(
        metric_type=metric_type,
    )
    metrics_artifacts = {
        k: v
        for k, v in metrics_artifacts.items()
        if k in metrics.Measurement.__annotations__.keys()
    }
    return metrics.Measurement(**metrics_artifacts)


@pytest.fixture
def batch_data(request: FixtureRequest) -> model.BatchData:
    """Create a batch data instance."""
    network_architecture = request.param.get("network_architecture")

    network_artifacts = w2v_fixtures.get_network_artifacts(
        network_architecture
    )
    idx_to_vocabulary = network_artifacts.get("idx_to_vocabulary")

    torch.manual_seed(27)
    word_idx = torch.randint(
        low=0, high=network_artifacts.get("vocabulary_size") - 1, size=(10,)
    )

    words = np.array(idx_to_vocabulary.get(int(idx)) for idx in word_idx)

    return model.BatchData(word_idx, words)
