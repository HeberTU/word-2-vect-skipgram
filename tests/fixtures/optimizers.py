# -*- coding: utf-8 -*-
"""Optimizer fixture module.

Created on: 9/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from word2vect.ml import models


def get_optimizer_config(
    model_type: models.ModelType,
) -> models.OptimizerConfig:
    """Get optimizer for the provided model type.

    Args:
        model_type: model type.

    Returns:
        optimizer_config: optimizer configuration.
    """
    _implementations = {
        models.ModelType.WORD2VECT: {
            "learning_rate": 0.02,
            "weight_decay": 0,
            "momentum": 0,
        }
    }

    config_params = _implementations.get(model_type, None)

    return models.OptimizerConfig(
        learning_rate=config_params.get("learning_rate"),
        weight_decay=config_params.get("weight_decay"),
        momentum=config_params.get("momentum"),
    )
