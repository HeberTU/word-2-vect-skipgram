# -*- coding: utf-8 -*-
"""Interface loss function module.

This module implement the abstract loss function class, which is used to define
the common interface across the different loss function implementations.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import torch


class LossFunction(ABC):
    """Loss function interface."""

    @abstractmethod
    def compute(
        self, result: Result, ground_truth: GroundTruth
    ) -> torch.Tensor:
        """Compute the loss based on network results and ground truth labels.

        Args:
            result: Network results.
            ground_truth: Ground truth.

        Returns:
            loss: loss results.
        """
        raise NotImplementedError()


@dataclass
class Result:
    """Data structure used to store different net outputs."""

    prediction: torch.Tensor
    log_prob: torch.Tensor
    measures: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class GroundTruth:
    """Data structure used to store the ground truth components."""

    target: torch.Tensor


@dataclass
class TrainingStats:
    """Data structure to store the training stats."""

    loss: List[torch.Tensor] = field(default_factory=list)
    prediction: List[torch.Tensor] = field(default_factory=list)
    target: List[torch.Tensor] = field(default_factory=list)
    log_every_n_steps: int = 500

    def update(
        self,
        loss: torch.Tensor,
        prediction: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> None:
        """Store the given statistics.

        Args:
            loss: calculated loss.
            prediction: network predictions.
            target: ground truth labels.

        """
        self.loss.append(loss.unsqueeze(dim=0))
        self.prediction.append(prediction)
        if target is not None:
            self.target.append(target)

    def flush(self):
        """Clear all stored statistics."""
        self.loss.clear()
        self.prediction.clear()
        self.target.clear()

    def get_training_stats(self) -> Iterable[Tuple[str, List[torch.Tensor]]]:
        """Iterate in lazy mode over all training statistics.

        Yields:
            loss: calculated loss.
            prediction: network predictions.
            target: ground truth labels.
        """
        yield "loss", self.loss
        yield "learning_prediction", self.prediction
        if len(self.target):
            yield "learning_target", self.target
