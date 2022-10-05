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
from torch.utils.data import DataLoader

import tests.fixtures as w2v_fixtures
from word2vect.ml import (
    algorithms,
    loss_functions,
    metrics,
    models,
    networks,
    tracker,
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


def get_model_definition(
    model_type: models.ModelType,
    network_architecture: networks.NetworkArchitecture,
) -> models.ModelDefinition:
    """Define model based on model type and network architecture.

    Args:
        model_type: model type.
        network_architecture: network architecture.

    Returns:
        model_definition: model definition data structure.
    """
    # Model config
    model_artifacts = w2v_fixtures.get_model_artifacts(model_type)
    model_config = w2v_fixtures.get_model_config(
        model_type=model_type,
        model_artifacts=model_artifacts,
    )

    # Network config
    network_artifacts = w2v_fixtures.get_network_artifacts(
        network_architecture
    )
    network_config = w2v_fixtures.get_network_config(
        network_architecture=network_architecture,
        network_artifacts=network_artifacts,
    )

    # Metrics config
    metrics_config = w2v_fixtures.get_metrics_config(model_type=model_type)

    # Loss function config
    loss_function_config = w2v_fixtures.get_loss_function_config(
        model_type=model_type
    )

    # Optimizer configuration
    optimizer_config = w2v_fixtures.get_optimizer_config(model_type=model_type)

    # Model definition
    model_definition = models.ModelDefinition(
        model_config=model_config,
        network_config=network_config,
        metrics_config=metrics_config,
        loss_function_config=loss_function_config,
        optimizer_config=optimizer_config,
    )

    return model_definition


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

    network_config = w2v_fixtures.get_network_config(
        network_architecture=network_architecture,
        network_artifacts=network_artifacts,
    )

    network_factory = networks.NetworkFactory(network_config=network_config)

    network = network_factory.create()

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
def batch_data(request: FixtureRequest) -> models.BatchData:
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

    words = np.array([idx_to_vocabulary.get(int(idx)) for idx in word_idx])

    return models.BatchData(word_idx, words)


@pytest.fixture
def model(request: FixtureRequest) -> models.NNModel:
    """Create a model instance."""
    # Get types.
    model_type = request.param.get("model_type")
    network_architecture = request.param.get("network_architecture")

    model_definition = get_model_definition(
        model_type=model_type, network_architecture=network_architecture
    )

    return models.ModelFactory(model_definition=model_definition).create()


@pytest.fixture()
def training_tracker(request: FixtureRequest) -> Type[tracker.TrainingTracker]:
    """Training tracker."""
    return w2v_fixtures.get_training_tracker(request.param)


@pytest.fixture()
def dataset(request: FixtureRequest) -> DataLoader:
    """Word2Vect Dataset."""
    dataset_size = request.param.get("dataset_size")
    batch_size = request.param.get("batch_size")

    dataset = w2v_fixtures.get_dataset(
        dataset_size=dataset_size,
        batch_size=batch_size,
    )

    return dataset


@pytest.fixture()
def data_loader(request: FixtureRequest) -> DataLoader:
    """Pytorch data loader."""
    batch_size = request.param.get("batch_size")
    dataset_size = request.param.get("dataset_size")

    data_loader = w2v_fixtures.get_data_loader(
        dataset_size=dataset_size,
        batch_size=batch_size,
    )
    return data_loader


@pytest.fixture()
def algorithm(request: FixtureRequest) -> Type[algorithms.Algorithm]:
    """Single network algorithm."""
    model_type = request.param.get("model_type")
    network_architecture = request.param.get("network_architecture")

    model_definition = get_model_definition(
        model_type=model_type, network_architecture=network_architecture
    )

    model = models.ModelFactory(model_definition=model_definition).create()

    network_artifacts = w2v_fixtures.get_network_artifacts(
        network_architecture
    )
    return model, network_artifacts
