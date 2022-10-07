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

    def __init__(
        self, words: List[int], batch_size: int, threshold: float = 1e-5
    ):
        """Initialize a word2vect dataset.

        Args:
            words: word in raw data format.
            batch_size: number of samples per batch.
            threshold: subsampling threshold
        """
        self.words = words
        self.batch_size = batch_size
        self.threshold = threshold
        self.train_words = self._subsampling()
        self.train_words = self._fit_words_to_batch_size()

    def __len__(self):
        """Get the number of train words."""
        return len(self.train_words)

    def __getitem__(self, idx: int) -> int:
        """Get data item.

        Args:
            item: index.

        Returns:
            word
        """
        return self.train_words[idx]

    def _subsampling(self) -> List[int]:
        """Create a subsampling from the original set of words.

        Args:
            threshold: parameter use to increase the drop rate of high
            frequent words.

        Returns:
            train_words: words use for training.
        """
        total_count = len(self.words)
        word_counts = Counter(self.words)

        self.freqs = {
            word: count / total_count for word, count in word_counts.items()
        }

        self.p_drop = {
            word: max(1 - np.sqrt(self.threshold / self.freqs[word]), 0)
            for word in word_counts
        }

        random.seed(27)
        train_words = [
            word
            for word in self.words
            if random.random() < (1 - self.p_drop[word])
        ]

        return train_words

    def _fit_words_to_batch_size(self) -> List[int]:
        """Select the maximum number of whole batches.

        Returns:
            train_words: words use for training.
        """
        n_batches = len(self.train_words) // self.batch_size
        return self.train_words[: n_batches * self.batch_size]
