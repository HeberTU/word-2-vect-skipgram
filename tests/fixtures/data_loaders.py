# -*- coding: utf-8 -*-
"""Data Loader Fixtures.

Created on: 19/9/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import random
import string

import numpy as np
from torch.utils.data import DataLoader

from word2vect.data import loaders
from word2vect.ml import networks


def get_vocabulary_space(vocabulary_size: int = 20) -> networks.Vocabulary:
    """Create the vocabulary space as a Vocabulary data class instance.

    Args:
        vocabulary_size: Number of unique tokens in the vocabulary.

    Returns:
        vocabulary: Data class containing the vocabulary specs.
    """
    vocabulary_to_idx = {
        char: idx
        for idx, char in enumerate(string.ascii_letters[:vocabulary_size])
    }
    idx_to_vocabulary = {idx: char for char, idx in vocabulary_to_idx.items()}

    return networks.Vocabulary(
        size=vocabulary_size,
        vocabulary_to_idx=vocabulary_to_idx,
        idx_to_vocabulary=idx_to_vocabulary,
    )


def get_dataset(
    dataset_size: int = 20, batch_size: int = 5
) -> loaders.W2VDataset:
    """Get torch data loader.

    Args:
        dataset_size: Number of words in the dataset.
        batch_size: number of samples per batch.
    Returns:
        w2vdataset: Word-2-Vect dataset.
    """
    vocabulary = get_vocabulary_space()

    probs = [2**x for x in range(vocabulary.size)]
    probs = [x / sum(probs) for x in probs]

    random.seed(27)

    words = np.random.choice(
        a=list(vocabulary.idx_to_vocabulary.keys()),
        size=dataset_size,
        replace=True,
        p=probs,
    )

    return loaders.W2VDataset(
        words=words, batch_size=batch_size, threshold=1 / vocabulary.size
    )


def get_data_loader(dataset_size: int = 20, batch_size: int = 5) -> DataLoader:
    """Get pytorch data loader.

    Args:
        dataset_size: Number of words in the dataset.
        batch_size: number of samples per batch.

    Returns:
        data_loader: An instance of pytorch's data loader.
    """
    return DataLoader(
        dataset=get_dataset(dataset_size, batch_size),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=loaders.skipgram_collate,
    )
