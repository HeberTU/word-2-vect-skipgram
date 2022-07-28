# -*- coding: utf-8 -*-
"""F1 score implementation.

It is useful to measure the performance of classification algorithms.

Created on: 28/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import dataclasses
from typing import (
    Any,
    Dict,
)

from sklearn import metrics as sklearn_metrics

from word2vect.ml import loss_functions
from word2vect.ml.metrics import interface


@dataclasses.dataclass
class F1Score(interface.Metric):
    """Data structure for storing the F1 score during run time."""

    metric_type: interface.MetricType = interface.MetricType.F1
    _metric_values: interface.MetricValues = dataclasses.field(
        default_factory=lambda: interface.MetricValues()
    )
    params: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"average": "macro"}
    )

    def measure(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
    ) -> interface.Measurement:
        """Measure model performance using F1 score.

        Args:
            result: prediction set.
            ground_truth: ground truth values.

        Returns:
            measurement: F1-score's measurement instance.
        """
        f1_score = sklearn_metrics.f1_score(
            y_true=ground_truth.target, y_pred=result.prediction, **self.params
        )

        measurement = interface.Measurement(
            value=f1_score, batch_size=len(result.prediction)
        )

        return measurement
