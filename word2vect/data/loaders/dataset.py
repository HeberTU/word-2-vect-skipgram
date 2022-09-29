# -*- coding: utf-8 -*-
"""This module contains the word2vect data set implementation.

Created on: 29/9/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import random
from collections import Counter
from typing import List

import numpy as np
from torch.utils.data import Dataset


class W2VDataset(Dataset):
    """This class represent a map from keys to data samples."""

    def __init__(self, words: List[int]):
        """Initialize a word2vect dataset.

        Args:
            words: word in raw data format.
        """
        self.words = words
        self.train_words = self._subsampling()

    def __getitem__(self, item: int):
        """Get data item.

        Args:
            item: index.

        Returns:
            data.
        """
        pass

    def _subsampling(self, threshold: float = 1e-5) -> List[int]:
        """Create a subsampling from the original set of words.

        Args:
            threshold: parameter use to increase the drop rate of high
            frequent words.

        Returns:
            train_words:
        """
        total_count = len(self.words)
        word_counts = Counter(self.words)

        freqs = {
            word: count / total_count for word, count in word_counts.items()
        }

        p_drop = {
            word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts
        }

        train_words = [
            word for word in self.words if random.random() < (1 - p_drop[word])
        ]

        return train_words
