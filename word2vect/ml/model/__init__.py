# -*- coding: utf-8 -*-
"""Model Library.

Created on: 23/6/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from .model_factory import ModelType
from .model_interface import (
    BatchData,
    ModelConfig,
)
from .word2vect_model import Word2VectModel

__all__ = [
    "Word2VectModel",
    "BatchData",
    "ModelConfig",
    "ModelType",
]
