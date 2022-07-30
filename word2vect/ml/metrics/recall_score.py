# -*- coding: utf-8 -*-
"""Recall score implementation.

Created on: 30/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import dataclasses
from typing import (
    Any,
    Dict,
    Optional,
)

from sklearn import metrics as sklearn_metrics

from word2vect.ml import loss_functions
from word2vect.ml.metrics import interface


@dataclasses.dataclass
class RecallScore(interface.Metric):
    """Data structure for storing the recall score during run time."""

    metric_type: interface.MetricType = interface.MetricType.RECALL
    _metric_values: interface.MetricValues = dataclasses.field(
        default_factory=lambda: interface.MetricValues()
    )
    params: Optional[Dict[str, Any]] = None

    def measure(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
    ) -> interface.Measurement:
        """Measure model performance using Recall score.

        Args:
            result: prediction set.
            ground_truth: ground truth values.

        Returns:
            measurement: Precision's measurement instance.
        """
        self.params = self.params if self.params else {"average": "macro"}

        f1_score = sklearn_metrics.recall_score(
            y_true=ground_truth.target, y_pred=result.prediction, **self.params
        )

        measurement = interface.Measurement(
            value=f1_score, batch_size=len(result.prediction)
        )

        return measurement
