# -*- coding: utf-8 -*-
"""This module test the skipgram architecture.

Created on: 18/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Type

import pytest
import torch
from torch import nn

from word2vect.ml import (
    models,
    networks,
)


@pytest.mark.parametrize(
    "network",
    [{"network_architecture": networks.NetworkArchitecture.SKIPGRAM}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_data",
    [{"network_architecture": networks.NetworkArchitecture.SKIPGRAM}],
    indirect=True,
)
def test_forward_output_shape(
    network: Type[nn.Module],
    batch_data: models.BatchData,
) -> None:
    """This function test that the output shapes are right."""
    network.eval()

    with torch.no_grad():
        outout = network(batch_data)

    assert outout.shape[0] == 10
    assert outout.shape[1] == network.features.vocabulary.size


@pytest.mark.parametrize(
    "network",
    [{"network_architecture": networks.NetworkArchitecture.SKIPGRAM}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_data",
    [{"network_architecture": networks.NetworkArchitecture.SKIPGRAM}],
    indirect=True,
)
def test_forward_output_probs(
    network: Type[nn.Module],
    batch_data: models.BatchData,
) -> None:
    """This function test that the output shapes are right."""
    network.eval()

    with torch.no_grad():
        outout = network(batch_data)
    assert float(outout.exp().sum()) == pytest.approx(10, 0.0001)
