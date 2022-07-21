# -*- coding: utf-8 -*-
"""Loss function factory module.

The factory module is a creational design pattern that provides an interface
for creating loss functions.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import enum


class LossFunctionType(enum.Enum):
    """Available loss functions types."""

    INTERFACE = enum.auto()
    NLLLOSS = enum.auto()
