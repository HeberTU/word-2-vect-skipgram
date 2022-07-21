# -*- coding: utf-8 -*-
"""Loss function factory module.

The factory module is a creational design pattern that provides an interface
for creating loss functions.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import enum
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Optional,
)

from word2vect.ml.loss_functions import negative_log_likelihood


class LossFunctionType(enum.Enum):
    """Available loss functions types."""

    INTERFACE = enum.auto()
    NLLLOSS = enum.auto()


@dataclass(frozen=True)
class LossFunctionConfig:
    """Data structure to store loss function config."""

    loss_function_type: LossFunctionType
    params: Optional[Dict[str, Any]] = None


class LossFunctionFactory:
    """Loss Function Factory class.

    This is a creational class used to instantiate the different loss function
    implementations.
    """

    def __init__(self, loss_function_config: LossFunctionConfig) -> None:
        """Instantiate the loss function factory.

        Args:
            loss_function_config: Loss function configuration.
        """
        self._config = loss_function_config
        self._loss_functions = {
            LossFunctionType.NLLLOSS: negative_log_likelihood.NLLLoss
        }

    def create(self, loss_function_type: LossFunctionType):
        """Create the loss function.

        Args:
            loss_function_type: loss function type.

        Returns:
            loss_function: loss function instance.
        """
        loss_function = self._loss_functions.get(loss_function_type, None)

        if loss_function is None:
            raise NotImplementedError(f"{loss_function_type} not implemented.")

        return loss_function(self._config.params)
