# -*- coding: utf-8 -*-
"""This module test the Word2Vect Model.

Created on: 9/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pytest
import torch

from word2vect.ml import (
    loss_functions,
    metrics,
    models,
    networks,
    tracker,
)

TEST_PARAMS = [
    {
        "model_type": models.ModelType.WORD2VECT,
        "network_architecture": networks.NetworkArchitecture.SKIPGRAM,
        "loss_artifacts_type": loss_functions.LossFunctionType.NLLLOSS,
    },
]


@pytest.mark.unit
@pytest.mark.parametrize("model", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("batch_data", TEST_PARAMS, indirect=True)
def test_word2vect_forward_step_serve(
    model: models.Word2VectModel,
    batch_data: models.BatchData,
) -> None:
    """Test model's forward method at serving stage."""
    predictions = model.forward(
        batch_data=batch_data, stage=tracker.Stage.SERVE
    )

    assert not model.network.training
    assert batch_data.word_idx.shape[0] == predictions.shape[0]
    assert model.network.embeddings.num_embeddings == predictions.shape[1]


@pytest.mark.unit
@pytest.mark.parametrize("model", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("batch_data", TEST_PARAMS, indirect=True)
def test_word2vect_forward_step_train(
    model: models.Word2VectModel,
    batch_data: models.BatchData,
) -> None:
    """Test model's forward method at trainign stage."""
    predictions = model.forward(
        batch_data=batch_data, stage=tracker.Stage.TRAIN
    )

    assert model.network.training
    assert batch_data.word_idx.shape[0] == predictions.shape[0]
    assert model.network.embeddings.num_embeddings == predictions.shape[1]


@pytest.mark.unit
@pytest.mark.parametrize("model", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("batch_data", TEST_PARAMS, indirect=True)
def test_word2vect_get_model_result(
    model: models.Word2VectModel,
    batch_data: models.BatchData,
) -> None:
    """Test model's get_model_result method."""
    predictions = model.forward(
        batch_data=batch_data, stage=tracker.Stage.TRAIN
    )

    result = model.get_model_result(predictions)
    assert isinstance(result, loss_functions.Result)
    assert result.prediction.shape[0] == batch_data.word_idx.shape[0]
    assert result.log_prob.shape[0] == batch_data.word_idx.shape[0]
    assert result.log_prob.shape[1] == model.network.embeddings.num_embeddings


@pytest.mark.unit
@pytest.mark.parametrize("model", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("batch_data", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("ground_truth", TEST_PARAMS, indirect=True)
def test_word2vect_learn(
    model: models.Word2VectModel,
    batch_data: models.BatchData,
    ground_truth: loss_functions.GroundTruth,
) -> None:
    """Test model's learn method."""
    predictions = model.forward(
        batch_data=batch_data, stage=tracker.Stage.TRAIN
    )

    result = model.get_model_result(predictions)

    _ = model.learn(result=result, ground_truth=ground_truth)

    for params in model.network.parameters():
        assert not (params.grad == torch.zeros(params.shape)).all().item()


@pytest.mark.unit
@pytest.mark.parametrize("model", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("batch_data", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("ground_truth", TEST_PARAMS, indirect=True)
def test_word2vect_evaluate(
    model: models.Word2VectModel,
    batch_data: models.BatchData,
    ground_truth: loss_functions.GroundTruth,
) -> None:
    """Test model's evaluate method."""
    predictions = model.forward(
        batch_data=batch_data, stage=tracker.Stage.TRAIN
    )

    result = model.get_model_result(predictions)

    loss = model.learn(result=result, ground_truth=ground_truth)

    assert_metrics_are_empty(model_metrics=model.model_metrics)

    model.evaluate(result, ground_truth, loss)

    assert_metrics_are_not_empty(model_metrics=model.model_metrics)


def assert_metrics_are_empty(model_metrics: metrics.ModelMetrics) -> None:
    """Test tha each component in Metrics class, if present, is empty."""
    assert len(model_metrics.optimizing_metric.metric_values.values) == 0
    assert len(model_metrics.optimizing_metric.metric_values.batch_sizes) == 0

    if model_metrics.secondary_metrics is not None:
        for _, secondary_metric in model_metrics.secondary_metrics.items():
            assert len(secondary_metric.metric_values.values) == 0
            assert secondary_metric.metric_values.batch_sizes == []


def assert_metrics_are_not_empty(model_metrics: metrics.ModelMetrics) -> None:
    """Test tha each component in Metrics class, if present, is not empty."""
    assert len(model_metrics.optimizing_metric.metric_values.values) > 0
    assert len(model_metrics.optimizing_metric.metric_values.batch_sizes) > 0

    if model_metrics.secondary_metrics is not None:
        for _, secondary_metric in model_metrics.secondary_metrics.items():
            assert len(secondary_metric.metric_values.values) > 0
            assert len(secondary_metric.metric_values.batch_sizes) > 0
