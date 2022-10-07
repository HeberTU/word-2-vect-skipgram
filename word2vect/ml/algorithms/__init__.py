# -*- coding: utf-8 -*-
"""Algorithms library.

Created on: 17/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from .algorithm import Algorithm
from .algorithm_factory import (
    AlgorithmConfig,
    AlgorithmFactory,
    AlgorithmType,
)
from .network_algorithm import NetworkAlgorithm

__all__ = [
    "Algorithm",
    "AlgorithmConfig",
    "AlgorithmType",
    "AlgorithmFactory",
    "NetworkAlgorithm",
]
