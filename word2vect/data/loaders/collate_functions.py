# -*- coding: utf-8 -*-
"""This module contains all the colleta functions for data loaders.

Created on: 3/10/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Tuple,
)

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from word2vect.ml import (
    loss_functions,
    models,
)


def skipgram_collate(
    data: Any,
) -> Tuple[models.BatchData, loss_functions.GroundTruth]:
    """Create a batch data for training a skipgram model.

    Args:
        data: single data batch.

    Returns:
        batch: features.
        ground_truth: target variables.
    """
    words_batch = default_collate(data)
    features = torch.Tensor()
    target = torch.Tensor()
    for i in range(len(words_batch)):
        feat_batch = words_batch[i]
        target_batch = get_target(words_batch, i)

        feat_batch = feat_batch.repeat(len(target_batch))

        features = torch.cat(tensors=(features, feat_batch))
        target = torch.cat(tensors=(target, target_batch))

    batch = models.BatchData(word_idx=features)

    ground_truth = loss_functions.GroundTruth(target=target)

    return batch, ground_truth


def get_target(
    words_batch: torch.Tensor, idx: int, window_size: int = 5
) -> torch.Tensor:
    """Get target values using the skip-gram architecture.

     For each word in the text, we want to define a surrounding context and
     grab all the words in a window around that word.

    Args:
        words_batch: Batch of words used as features.
        idx: curren index.
        window_size: size of the training window.

    Returns:
        window: word window around target.
    """
    window = torch.Tensor()
    r = np.random.randint(1, window_size + 1)

    if idx - r < 0:
        window = torch.cat(tensors=(window, words_batch[0:idx]))
    else:
        start_idx = idx - r
        window = torch.cat(tensors=(window, words_batch[start_idx:idx]))

    start_idx = idx + 1
    end_idx = idx + r + 1
    window = torch.cat(tensors=(window, words_batch[start_idx:end_idx]))

    return window
