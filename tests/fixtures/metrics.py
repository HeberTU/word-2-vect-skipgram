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

from word2vect.ml import (
    metrics,
    models,
)


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


def get_metrics_config(model_type: models.ModelType) -> metrics.MetricsConfig:
    """Get metrics configuration for model.

    This function temporarily only accepts the Word2Vect config.

    Returns:
        metrics_config: metrics configuration.
    """
    params = {"average": "macro"}

    _implementation = {
        models.ModelType.WORD2VECT: {
            "optimizing_metric": metrics.MetricType.F1,
            "secondary_metrics": [
                metrics.MetricType.PRECISION,
                metrics.MetricType.RECALL,
            ],
        },
    }

    _metrics = _implementation.get(model_type, None)

    return metrics.MetricsConfig(
        optimizing_metric=metrics.MetricConfig(
            metric_type=_metrics.get("optimizing_metric"),
            params=params,
        ),
        secondary_metrics=[
            metrics.MetricConfig(metric_type=metric_type, params=params)
            for metric_type in _metrics.get("secondary_metrics")
        ],
    )
