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
    ModelMetrics,
)
from word2vect.ml.metrics.precision_score import PrecisionScore
from word2vect.ml.metrics.recall_score import RecallScore


@dataclass(frozen=True)
class MetricsConfig:
    """Data structure to store model's metrics set-up."""

    optimizing_metric: MetricConfig
    secondary_metrics: Optional[List[MetricConfig]] = None


@dataclass(frozen=True)
class MetricConfig:
    """Data structure for storing metric configuration."""

    metric_type: MetricType
    params: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        """Get metric name."""
        return self.metric_type.value


class AverageStrategy(enum.Enum):
    """Available metrics reduction strategies."""

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"


class MetricFactory:
    """Metric Factory class.

    This is a creational class used to instantiate the different loss function
    implementations.
    """

    def __init__(self, metric_config: MetricConfig) -> MetricFactory:
        """Instantiate the metric factory.

        Args:
            params: Metric configuration params.
        """
        self._metric_config = metric_config
        self._metrics = {
            MetricType.F1: F1Score,
            MetricType.PRECISION: PrecisionScore,
            MetricType.RECALL: RecallScore,
        }

    def create(self) -> Type[Metric]:
        """Create a metric using the init config.

        Returns:
            metric.
        """
        metric = self._metrics.get(self._metric_config.metric_type, None)

        if metric is None:
            raise NotImplementedError(
                f"{self._metric_config.metric_type} not implemented."
            )

        return metric(self._metric_config.params)


class ModelMetricsFactory:
    """Model Metrics Factory class.

    This is a creational class used to instantiate the different loss function
    implementations.
    """

    def __init__(self, metrics_config: MetricsConfig) -> ModelMetricsFactory:
        """Instantiate a model metrics factory.

        Args:
            metrics_config: metrics config.
        """
        self._config = metrics_config

    def create(self) -> ModelMetrics:
        """Create a model metrics instance using the init config.

        Returns:
            model_metrics: model metrics.
        """
        optimizing_metric = MetricFactory(
            metric_config=self._config.optimizing_metric
        ).create()

        if self._config.secondary_metrics is not None:
            secondary_metrics = {}
            for metric_config in self._config.secondary_metrics:
                secondary_metrics[metric_config.name] = MetricFactory(
                    metric_config=metric_config
                ).create()
        else:
            secondary_metrics = None

        return ModelMetrics(
            optimizing_metric=optimizing_metric,
            secondary_metrics=secondary_metrics,
        )
