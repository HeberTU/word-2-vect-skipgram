# -*- coding: utf-8 -*-
"""Negative log likelihood loss implementation.

It is useful to train a classification problem with `C` classes.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import torch

import word2vect.ml.loss_functions.interface as interface


class NLLLoss(interface.LossFunction):
    """Negative log likelihood loss implementation."""

    def __init__(self):
        """Initialize the class."""
        self._torch_nlloss = torch.nn.NLLLoss()

    def __repr__(self) -> str:
        """Create the string representation."""
        return "NLLLoss()"

    def compute(
        self, result: interface.Result, ground_truth: interface.GroundTruth
    ) -> torch.Tensor:
        """Compute the negative log likelihood loss.

        Args:
            result: network results.
            ground_truth: Ground Truth data.

        Returns:
            loss: negative log likelihood loss
        """
        return self._torch_nlloss(result.prediction, ground_truth.target)
