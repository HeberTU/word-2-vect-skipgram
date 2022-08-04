# -*- coding: utf-8 -*-
"""Interface metrics module.

This module implement the abstract metric class, which is used to define
the common interface across the different metrics implementations.

Created on: 27/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

import enum
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
)

import numpy as np

from word2vect.ml import loss_functions


@dataclass(frozen=True)
class ModelsRepr:
    """Model Representation."""

    raw_repr: str
    hashed_repr: str


class MetricType(enum.Enum):
    """Available metrics."""

    INTERFACE = enum.auto()
    F1 = enum.auto()
    PRECISION = enum.auto()
    RECALL = enum.auto()


@dataclass
class ModelMetrics:
    """Data Structure to store model metrics."""

    optimizing_metric: Type[Metric]
    secondary_metrics: Optional[Dict[str, Type[Metric]]] = None
    log_every_n_steps: int = 100
    _models_repr: ModelsRepr = field(init=False, repr=False)

    def get_metrics(self) -> Iterable[Metric]:
        """Retrieve all metrics in a generator format.

        Returns:
            metrics: metric instances.
        """
        yield self.optimizing_metric

        if self.secondary_metrics:
            for secondary_metric in self.secondary_metrics.values():
                yield secondary_metric

    def update(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
    ) -> None:
        """Update all model metrics.

        Args:
            result: prediction set.
            ground_truth: ground truth values.

        Returns:
            None.
        """
        for metric in self.get_metrics():
            metric.metric_values = metric.measure(result, ground_truth)

    def flush(self) -> None:
        """Reset all model metrics."""
        for metric in self.get_metrics():
            metric.metric_values.flush()


@dataclass
class Metric(ABC):
    """Data structure for storing the running metrics."""

    metric_type: MetricType
    _metric_values: MetricValues
    params: Optional[Dict[str, Any]] = None

    @property
    def metric_values(self) -> MetricValues:
        """Retrieve the metric values."""
        return self._metric_values

    @property
    def average_value(self) -> float:
        """Retrieve the average metric value."""
        return self._metric_values.average_value

    @metric_values.setter
    def metric_values(self, measurement: Measurement) -> None:
        self._metric_values.update(measurement)

    @abstractmethod
    def measure(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
    ) -> Measurement:
        """Measure model performance.

        Args:
            result: prediction set.
            ground_truth: ground truth values.


        Returns:
            measurement: model performance measure.
        """
        raise NotImplementedError()


@dataclass
class MetricValues:
    """Data structure to store the metrics values."""

    values: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)

    def update(self, measurement: Measurement) -> None:
        """Append measurement into metrics history.

        Args:
            measurement: running measurements.

        Returns:
            None
        """
        self.values.append(measurement.value)
        self.batch_sizes.append(measurement.batch_size)

    @property
    def average_value(self) -> float:
        """Calculate average measurement value."""
        return round(
            sum(np.array(self.values) * np.array(self.batch_sizes))
            / sum(self.batch_sizes),
            2,
        )

    def flush(self) -> None:
        """Reset the metric values."""
        self.values = []
        self.batch_sizes = []


@dataclass
class Measurement:
    """Data structure to store metrics measures during run."""

    value: float
    batch_size: int
