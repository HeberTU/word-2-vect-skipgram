# -*- coding: utf-8 -*-
"""Unit test for dataset utils.

Created on: 30/9/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from collections import Counter

import pytest
from torch.utils.data import DataLoader

from word2vect.data import loaders

TEST_PARAMS = [
    {"batch_size": 128, "dataset_size": 2000},
]


@pytest.mark.unit
@pytest.mark.parametrize("dataset", TEST_PARAMS, indirect=True)
def test_dataset_fits_batch_size(dataset: loaders.W2VDataset) -> None:
    """Test train words has been reduced to fit batch size."""
    assert len(dataset.train_words) % dataset.batch_size == 0


@pytest.mark.unit
@pytest.mark.parametrize("dataset", TEST_PARAMS, indirect=True)
def test_dataset_subsampling(dataset: loaders.W2VDataset) -> None:
    """Test subsampling method."""
    word_counts = Counter(dataset.words)

    train_word_counts_est = {
        word: (1 - dataset.p_drop[word]) * count
        for word, count in word_counts.items()
    }

    train_word_counts = Counter(dataset.train_words)

    differences = {
        word: counts / train_word_counts_est[word]
        for word, counts in train_word_counts.items()
    }

    ratio = sum(list(differences.values())) / len(list(differences.values()))

    assert int(round(ratio)) == 1


@pytest.mark.unit
@pytest.mark.parametrize("data_loader", TEST_PARAMS, indirect=True)
def test_skipgram_collate_fun(data_loader: DataLoader):
    """Test batch and target match in size."""
    batch, ground_truth = next(iter(data_loader))

    assert len(batch.word_idx) == len(ground_truth.target)
