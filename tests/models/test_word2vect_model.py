# -*- coding: utf-8 -*-
"""This module test the Word2Vect Model.

Created on: 9/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pytest

from word2vect.ml import (
    loss_functions,
    models,
    networks,
    tracker,
)

TEST_PARAMS = [
    {
        "model_type": models.ModelType.WORD2VECT,
        "network_architecture": networks.NetworkArchitecture.SKIPGRAM,
    },
]


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
