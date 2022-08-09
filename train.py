# -*- coding: utf-8 -*-
"""Trainign script.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from torch import nn

from word2vect.data.data_prep import (
    create_lookup_tables,
    load_data,
    preprocess,
)
from word2vect.ml import (
    loss_functions,
    metrics,
    models,
    networks,
)


def main():
    """Train model."""
    text = load_data("data/text8")
    words = preprocess(text)
    vocab_to_int, int_to_vocab = create_lookup_tables(words)

    model_definition = models.ModelDefinition(
        model_config=models.ModelConfig(
            model_type=models.ModelType.WORD2VECT,
            model_name="word2vect_v01",
            gradient_clipping_value=1,
        ),
        network_config=networks.NetworkConfig(
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
        ),
        metrics_config=metrics.MetricsConfig(
            optimizing_metric=metrics.MetricConfig(
                metric_type=metrics.MetricType.F1, params={"average": "macro"}
            ),
            secondary_metrics=[
                metrics.MetricConfig(
                    metric_type=metrics.MetricType.PRECISION,
                    params={"average": "macro"},
                ),
                metrics.MetricConfig(
                    metric_type=metrics.MetricType.RECALL,
                    params={"average": "macro"},
                ),
            ],
        ),
        loss_function_config=loss_functions.LossFunctionConfig(
            loss_function_type=loss_functions.LossFunctionType.NLLLOSS
        ),
    )

    model_factory = models.ModelFactory(model_definition=model_definition)
    word2vect_model = model_factory.create()

    return word2vect_model


if __name__ == "__main__":
    main()
