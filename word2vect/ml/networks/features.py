# -*- coding: utf-8 -*-
"""Features abstraction module.

Created on: 18/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Features:
    """features definition."""

    vocabulary: Vocabulary
    embedding_dim: int


@dataclass(frozen=True)
class Vocabulary:
    """Vocabulary definition."""

    size: int
    vocabulary_to_idx: Dict[str, int]
    idx_to_vocabulary: Dict[int, str]


@dataclass(frozen=True)
class Batch:
    """Batch configuration definition."""

    pass
