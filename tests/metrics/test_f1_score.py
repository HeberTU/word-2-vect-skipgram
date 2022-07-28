# -*- coding: utf-8 -*-
"""This module test the f1-score metric calculation.

Created on: 28/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pytest

from word2vect.ml import (
    loss_functions,
    metrics,
)

TEST_PARAMS = [
    {
        "loss_artifacts_type": loss_functions.LossFunctionType.INTERFACE,
        "metric_type": metrics.MetricType.F1,
    }
]


@pytest.mark.unit
@pytest.mark.parametrize("metric", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("result", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("ground_truth", TEST_PARAMS, indirect=True)
def test_f1_score_measure(
    metric: metrics.F1Score,
    result: loss_functions.Result,
    ground_truth: loss_functions.GroundTruth,
) -> None:
    """Test f1 score calculation."""
    measurement = metric.measure(result, ground_truth)

    assert measurement.value == pytest.approx(0.26666666)
    assert measurement.batch_size == 6
