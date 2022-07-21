# -*- coding: utf-8 -*-
"""Loss function library.

This library implements the interface and implementation onf loss functions.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from .interface import (
    GroundTruth,
    LossFunction,
    Result,
    TrainingStats,
)
from .loss_function_factory import LossFunctionType
from .negative_log_likelihood import NLLLoss

__all__ = [
    "TrainingStats",
    "LossFunction",
    "Result",
    "GroundTruth",
    "NLLLoss",
    "LossFunctionType",
]
