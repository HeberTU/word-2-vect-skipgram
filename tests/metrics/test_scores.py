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

TEST_PARAMS_F1 = [
    {
        "loss_artifacts_type": loss_functions.LossFunctionType.INTERFACE,
        "metric_type": metrics.MetricType.F1,
    },
]
TEST_PARAMS_PRECISION = [
    {
        "loss_artifacts_type": loss_functions.LossFunctionType.INTERFACE,
        "metric_type": metrics.MetricType.PRECISION,
    },
]
TEST_PARAMS_RECALL = [
    {
        "loss_artifacts_type": loss_functions.LossFunctionType.INTERFACE,
        "metric_type": metrics.MetricType.RECALL,
    },
]


@pytest.mark.unit
@pytest.mark.parametrize("metric", TEST_PARAMS_F1, indirect=True)
@pytest.mark.parametrize("result", TEST_PARAMS_F1, indirect=True)
@pytest.mark.parametrize("ground_truth", TEST_PARAMS_F1, indirect=True)
@pytest.mark.parametrize("measurement", TEST_PARAMS_F1, indirect=True)
def test_f1_score_measure(
    metric: metrics.F1Score,
    result: loss_functions.Result,
    ground_truth: loss_functions.GroundTruth,
    measurement: metrics.Measurement,
) -> None:
    """Test f1 score calculation."""
    result_measurement = metric.measure(result, ground_truth)

    assert result_measurement.value == pytest.approx(measurement.value)
    assert result_measurement.batch_size == measurement.batch_size


@pytest.mark.unit
@pytest.mark.parametrize("metric", TEST_PARAMS_PRECISION, indirect=True)
@pytest.mark.parametrize("result", TEST_PARAMS_PRECISION, indirect=True)
@pytest.mark.parametrize("ground_truth", TEST_PARAMS_PRECISION, indirect=True)
@pytest.mark.parametrize("measurement", TEST_PARAMS_PRECISION, indirect=True)
def test_precision_score_measure(
    metric: metrics.PrecisionScore,
    result: loss_functions.Result,
    ground_truth: loss_functions.GroundTruth,
    measurement: metrics.Measurement,
) -> None:
    """Test precision score calculation."""
    result_measurement = metric.measure(result, ground_truth)

    assert result_measurement.value == pytest.approx(measurement.value)
    assert result_measurement.batch_size == measurement.batch_size


@pytest.mark.unit
@pytest.mark.parametrize("metric", TEST_PARAMS_RECALL, indirect=True)
@pytest.mark.parametrize("result", TEST_PARAMS_RECALL, indirect=True)
@pytest.mark.parametrize("ground_truth", TEST_PARAMS_RECALL, indirect=True)
@pytest.mark.parametrize("measurement", TEST_PARAMS_RECALL, indirect=True)
def test_recall_score_measure(
    metric: metrics.RecallScore,
    result: loss_functions.Result,
    ground_truth: loss_functions.GroundTruth,
    measurement: metrics.Measurement,
) -> None:
    """Test precision score calculation."""
    result_measurement = metric.measure(result, ground_truth)

    assert result_measurement.value == pytest.approx(measurement.value)
    assert result_measurement.batch_size == measurement.batch_size
