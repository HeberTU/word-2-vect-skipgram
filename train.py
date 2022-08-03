# -*- coding: utf-8 -*-
"""Trainign script.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from torch import nn
from torch.optim import SGD

from word2vect.data.data_prep import (
    create_lookup_tables,
    load_data,
    preprocess,
)
from word2vect.ml import (
    loss_functions,
    metrics,
    model,
    networks,
)


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

    network = networks.NetworkFactory(network_config=network_config).create(
        network_architecture=networks.NetworkArchitecture.SKIPGRAM
    )

    f1_score = metrics.MetricFactory(
        metric_config=metrics.MetricConfig()
    ).create(metric_type=metrics.MetricType.F1)

    recall_score = metrics.MetricFactory(
        metric_config=metrics.MetricConfig()
    ).create(metric_type=metrics.MetricType.RECALL)

    precision_score = metrics.MetricFactory(
        metric_config=metrics.MetricConfig()
    ).create(metric_type=metrics.MetricType.PRECISION)

    model_metrics = metrics.ModelMetrics(
        optimizing_metric=f1_score,
        secondary_metrics={
            "recall_score": recall_score,
            "precision_score": precision_score,
        },
    )

    model_config = model.ModelConfig(
        model_type=model.ModelType.WORD2VECT,
        model_name="word2vect_v01",
        gradient_clipping_value=1,
    )

    optimizer = SGD(network.parameters(), lr=0.1, momentum=0.9)

    loss_function = loss_functions.LossFunctionFactory(
        loss_function_config=loss_functions.LossFunctionConfig()
    ).create(loss_function_type=loss_functions.LossFunctionType.NLLLOSS)

    word2vect_model = model.Word2VectModel(
        network=networks,
        model_metrics=model_metrics,
        model_config=model_config,
        optimizer=optimizer,
        loss_function=loss_function,
    )

    return word2vect_model


if __name__ == "__main__":
    main()
