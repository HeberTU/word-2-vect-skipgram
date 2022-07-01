# -*- coding: utf-8 -*-
"""Model configuration skelethon.

Created on: 1/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations
from typing import NewType, Optional, List, Dict, Union

from dataclasses import dataclass
from enum import Enum, auto

from torch import nn
import torch.optim as optim

ModelName = NewType("ModelType", str)
Activation = Union[nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.SELU, nn.Identity]
Criterion = Union[nn.NLLLoss]
Optimizer = Union[optim.Adam]


@dataclass(frozen=True)
class Model:
    """Model Definition."""
    model_config: ModelConfig
    network_architecture: NetworkArchitecture
    loss_function_config: LossFunctionConfig
    optimizer_config: OptimizerConfig


@dataclass(frozen=True)
class ModelConfig:
    """Model Configurations"""
    model_type: ModelType
    model_name: ModelName
    network_weights_path: Optional[str] = None


class ModelType(Enum):
    """Available model types"""
    SKIPGRAM: int = auto()
    NGRAM: int = auto()


@dataclass(frozen=True)
class NetworkArchitecture:
    """Neural Network Architecture."""
    features: Features
    hidden_layers: HiddenLayers
    outputs: Outputs


@dataclass(frozen=True)
class Features:
    """Model features configuration."""
    words: List[str]
    word_idx: List[int]
    vocabulary: Vocabulary


@dataclass(frozen=True)
class Vocabulary:
    """Word vocabulary."""
    words_to_idx: Dict[str, int]
    idx_to_word: Dict[int, str]


@dataclass(frozen=True)
class HiddenLayers:
    """Hidden layers definitions."""
    hidden: List[int]
    activation: Activation
    dropout: Optional[float] = None


@dataclass(frozen=True)
class Outputs:
    """Output Definitions."""
    number: int
    activation: Activation


@dataclass(frozen=True)
class LossFunctionConfig:
    """Loss function definition."""
    criterion: Criterion


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimization criterion definition."""
    optimizer: Optimizer
