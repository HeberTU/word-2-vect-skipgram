# -*- coding: utf-8 -*-
"""Networks interface, implementations and factory.

Created on: 18/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from .features import (
    Batch,
    Features,
    Vocabulary,
)
from .fully_connected import (
    HiddenLayers,
    OutputLayer,
    build_sequential_layers,
)
from .network_factory import (
    NetworkArchitecture,
    NetworkConfig,
    NetworkFactory,
)
from .skipgram import SkipGram

__all__ = [
    "Features",
    "Vocabulary",
    "Batch",
    "HiddenLayers",
    "OutputLayer",
    "NetworkArchitecture",
    "NetworkFactory",
    "NetworkConfig",
    "SkipGram",
    "build_sequential_layers",
]
