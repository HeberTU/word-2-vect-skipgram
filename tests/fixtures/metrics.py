# -*- coding: utf-8 -*-
"""Metrics fixtures.

Created on: 5/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Dict,
    List,
    Union,
)

from word2vect.ml import metrics


def get_metrics_artifacts(
    metric_type: metrics.MetricType,
) -> Dict[str, Union[List[int], int]]:
    """Create metrics artifacts.

    Args:
        metric_type: metric type.

    Returns:
        metric_artifacts: Dictionary containing all the necessary params.
    """
    params = {"average": "macro"}
    concrete_batch_size = 6

    _implementations = {
        metrics.MetricType.INTERFACE: {
            "value": 0.9,
            "batch_size": 512,
            "params": params,
        },
        metrics.MetricType.F1: {
            "value": 0.26666666,
            "batch_size": concrete_batch_size,
            "params": params,
        },
        metrics.MetricType.PRECISION: {
            "value": 0.33333333,
            "batch_size": concrete_batch_size,
            "params": params,
        },
        metrics.MetricType.RECALL: {
            "value": 0.2222222,
            "batch_size": concrete_batch_size,
            "params": params,
        },
    }

    metric_artifacts = _implementations.get(metric_type, None)

    return metric_artifacts
