# -*- coding: utf-8 -*-
"""Neural Network ports and adapters.

Created on: 1/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from torch import nn
from word2vect.ml.factory import ModelConfig

class NeuralNet(ABC):
    """Abstract class for Neural Networks"""

    def __init__(
            self,
            network: nn.Module,
            model_config: ModelConfig,
            loss_function: Any
    ) -> None:
        """Constructor will be overridden in concrete implementations."""
        self.network = network
        self.model_config = model_config
