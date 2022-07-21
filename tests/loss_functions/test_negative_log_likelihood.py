# -*- coding: utf-8 -*-
"""This module test the negative log likelihood loss.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pytest
import torch

import word2vect.ml.loss_functions as loss_functions

TEST_PARAMS = {"loss_artifacts_type": loss_functions.LossFunctionType.NLLLOSS}


@pytest.mark.unit
@pytest.mark.parametrize("result", [TEST_PARAMS], indirect=True)
@pytest.mark.parametrize("loss", [TEST_PARAMS], indirect=True)
@pytest.mark.parametrize("ground_truth", [TEST_PARAMS], indirect=True)
def test_negative_log_likelihood_loss_compute(
    result: loss_functions.Result,
    ground_truth: loss_functions.GroundTruth,
    loss: torch.Tensor,
) -> None:
    """Test negative log_likelihood loss calculation."""
    nll_loss = loss_functions.NLLLoss()
    resulting_loss = float(nll_loss.compute(result, ground_truth))
    assert loss == pytest.approx(resulting_loss, 0.0001)
