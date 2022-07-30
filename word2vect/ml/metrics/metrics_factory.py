# -*- coding: utf-8 -*-
"""Metrics factory module.

The module provides the constructor factory for the different metrics
implementations, as well as the configuration structures.


Created on: 27/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
)

from word2vect.ml.metrics.f1_score import F1Score
from word2vect.ml.metrics.interface import (
    Metric,
    MetricType,
)
from word2vect.ml.metrics.precision_score import PrecisionScore
from word2vect.ml.metrics.recall_score import RecallScore


@dataclass(frozen=True)
class MetricsConfig:
    """Data structure to store model's metrics set-up."""

    optimizing_metric: MetricConfig
    secondary_metrics: List[MetricConfig]


@dataclass(frozen=True)
class MetricConfig:
    """Data structure for storing metric configuration."""

    metric_type: MetricType
    params: Optional[Dict[str, Any]] = None


class AverageStrategy(enum.Enum):
    """Available metrics reduction strategies."""

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"


class MetricFactory:
    """Loss Function Factory class.

    This is a creational class used to instantiate the different loss function
    implementations.
    """

    def __init__(self, metric_config: MetricConfig) -> MetricFactory:
        """Instantiate the metric factory.

        Args:
            metric_config: Metric configuration params.
        """
        self._config = metric_config
        self._metrics = {
            MetricType.F1: F1Score,
            MetricType.PRECISION: PrecisionScore,
            MetricType.RECALL: RecallScore,
        }

    def create(self, metric_type: MetricType) -> Type[Metric]:
        """Create a metric.

        Args:
            metric_type: metric type.

        Returns:
            metric.
        """
        metric = self._metrics.get(metric_type, None)

        if metric is None:
            raise NotImplementedError(f"{metric_type} not implemented.")

        return metric(self._config.params)
