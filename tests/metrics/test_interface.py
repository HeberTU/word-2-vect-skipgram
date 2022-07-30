# -*- coding: utf-8 -*-
"""This module test the metrics interface helper data structures.

Created on: 28/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pytest

import word2vect.ml.metrics as metrics

TEST_PARAMS = {"metric_type": metrics.MetricType.INTERFACE}


@pytest.mark.unit
@pytest.mark.parametrize("measurement", [TEST_PARAMS], indirect=True)
def test_metric_values_update(
    metric_values: metrics.MetricValues, measurement: metrics.Measurement
) -> None:
    """Test that MetricValues update method append measurements."""
    assert len(metric_values.values) == 0
    assert len(metric_values.batch_sizes) == 0

    metric_values.update(measurement)

    assert len(metric_values.values) > 0
    assert len(metric_values.batch_sizes) > 0


@pytest.mark.unit
@pytest.mark.parametrize("measurement", [TEST_PARAMS], indirect=True)
def test_metric_values_average_value(
    metric_values: metrics.MetricValues, measurement: metrics.Measurement
) -> None:
    """Test that MetricValues average value property."""
    metric_values.update(measurement)

    measurement.value = 0.5

    metric_values.update(measurement)

    assert metric_values.average_value == 0.7


@pytest.mark.unit
@pytest.mark.parametrize("measurement", [TEST_PARAMS], indirect=True)
def test_metric_values_flush(
    metric_values: metrics.MetricValues, measurement: metrics.Measurement
) -> None:
    """Test that MetricValues flush method reset measurements."""
    metric_values.update(measurement)

    assert len(metric_values.values) > 0
    assert len(metric_values.batch_sizes) > 0

    metric_values.flush()

    assert len(metric_values.values) == 0
    assert len(metric_values.batch_sizes) == 0
