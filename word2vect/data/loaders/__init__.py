# -*- coding: utf-8 -*-
"""Data loaders library.

Created on: 23/6/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from .collate_functions import skipgram_collate
from .dataset import W2VDataset

__all__ = ["W2VDataset", "skipgram_collate"]
