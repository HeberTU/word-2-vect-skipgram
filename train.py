# -*- coding: utf-8 -*-
"""Trainign script.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import torch
from torch import nn

from word2vect.data.data_prep import (
    create_lookup_tables,
    load_data,
    preprocess,
)
from word2vect.ml import networks


def main():
    """Train model."""
    text = load_data("data/text8")
    words = preprocess(text)
    vocab_to_int, int_to_vocab = create_lookup_tables(words)

    network_config = networks.NetworkConfig(
        features=networks.Features(
            vocabulary=networks.Vocabulary(
                size=len(vocab_to_int),
                vocabulary_to_idx=vocab_to_int,
                idx_to_vocabulary=int_to_vocab,
            ),
            embedding_dim=300,
        ),
        hidden_layers=networks.HiddenLayers(
            hidden_dim=None, activation=None, dropout=None
        ),
        output_layer=networks.OutputLayer(activation=nn.LogSoftmax(dim=1)),
    )

    model = networks.NetworkFactory(network_config=network_config).create(
        network_architecture=networks.NetworkArchitecture.SKIPGRAM
    )

    x = torch.tensor([1])

    pred = model(x)

    return pred


if __name__ == "__main__":
    main()
