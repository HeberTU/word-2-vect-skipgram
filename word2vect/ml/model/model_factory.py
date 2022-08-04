# -*- coding: utf-8 -*-
"""Model factory module.

Created on: 2/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import enum


class ModelType(enum.Enum):
    """Available model types."""

    WORD2VECT = enum.auto()
