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
from scipy import stats
from torch.utils.data import Dataset


class W2VDataset(Dataset):
    """This class represent a map from keys to data samples."""

    def __init__(self, words: List[int], batch_size: int):
        """Initialize a word2vect dataset.

        Args:
            words: word in raw data format.
            batch_size: number of samples per batch.
        """
        self.words = words
        self.batch_size = self.batch_size
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

    def _subsampling(self, threshold: float = 1e-5) -> List[int]:
        """Create a subsampling from the original set of words.

        Args:
            threshold: parameter use to increase the drop rate of high
            frequent words.

        Returns:
            train_words: words use for training.
        """
        pr, _ = stats.kstest(
            rvs=self.words,
            cdf=stats.uniform(
                loc=int(np.mean(self.words)),
                scale=np.std(self.words),
            ).cdf,
        )

        if pr > 0.05:
            return self.words

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

    def _fit_words_to_batch_size(self) -> List[int]:
        """Select the maximum number of whole batches.

        Returns:
            train_words: words use for training.
        """
        n_batches = len(self.train_words) // self.batch_size
        return self.train_words[: n_batches * self.batch_size]
