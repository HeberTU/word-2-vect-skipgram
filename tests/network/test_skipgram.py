# -*- coding: utf-8 -*-
"""This module test the skipgram architecture.

Created on: 18/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pytest
import torch

from word2vect.ml import networks


@pytest.mark.parametrize(
    "skipgram",
    [
        {
            "vocabulary_size": 20,
            "embedding_dim": 10,
            "hidden_dim": [5],
            "dropout": 0.2,
        }
    ],
    indirect=True,
)
def test_forward_output_shape(skipgram: networks.SkipGram) -> None:
    """This function test that the output shapes are right."""
    x = torch.randint(
        low=0, high=skipgram.features.vocabulary.size - 1, size=(10,)
    )

    skipgram.eval()

    with torch.no_grad():
        outout = skipgram(x)

    assert outout.shape[0] == 10
    assert outout.shape[1] == skipgram.features.vocabulary.size


@pytest.mark.parametrize(
    "skipgram",
    [
        {
            "vocabulary_size": 20,
            "embedding_dim": 10,
            "hidden_dim": [5],
            "dropout": 0.2,
        }
    ],
    indirect=True,
)
def test_forward_output_probs(skipgram: networks.SkipGram) -> None:
    """This function test that the output shapes are right."""
    x = torch.randint(
        low=0, high=skipgram.features.vocabulary.size - 1, size=(10,)
    )

    skipgram.eval()

    with torch.no_grad():
        outout = skipgram(x)
    assert int(outout.exp().sum()) == 10
