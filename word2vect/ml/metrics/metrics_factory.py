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
)


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


class MetricType(enum.Enum):
    """Available metrics."""

    INTERFACE = enum.auto()
    F1 = enum.auto()
    PRECISION = enum.auto()
    RECALL = enum.auto()


class AverageStrategy(enum.Enum):
    """Available metrics reduction strategies."""

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
