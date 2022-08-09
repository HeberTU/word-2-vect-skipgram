# -*- coding: utf-8 -*-
"""Loss function fixtures.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Dict

import torch

from word2vect.ml import (
    loss_functions,
    models,
)


def get_loss_artifacts(
    loss_artifacts_type: loss_functions.LossFunctionType,
) -> Dict[str, torch.Tensor]:
    """Create loss function artifacts."""
    nllloss_predictions = torch.Tensor(
        [
            [-1.6913, -1.7227, -2.2291, -1.3133, -1.3449],
            [-2.7042, -3.7364, -1.7378, -3.1297, -0.3716],
            [-1.6004, -1.4860, -3.4763, -1.6272, -1.0656],
        ]
    )

    _implementations = {
        loss_functions.LossFunctionType.INTERFACE: {
            "loss": torch.tensor([1]),
            "prediction": torch.tensor([0, 1, 2, 0, 1, 2]),
            "target": torch.tensor([0, 2, 1, 0, 0, 1]),
        },
        loss_functions.LossFunctionType.NLLLOSS: {
            "loss": 1.8308,
            "prediction": nllloss_predictions,
            "target": torch.tensor([1, 0, 4]),
        },
    }

    loss_artifacts = _implementations.get(loss_artifacts_type, None)

    return loss_artifacts


def get_loss_function_config(
    model_type: models.ModelType,
) -> loss_functions.LossFunctionConfig:
    """Get loss functions configuration for model type.

    Args:
        model_type: model type.

    Returns:
        loss_function_config: los function configuration.
    """
    _implementations = {
        models.ModelType.WORD2VECT: {
            "loss_function_type": loss_functions.LossFunctionType.NLLLOSS
        }
    }

    config_params = _implementations.get(model_type, None)

    return loss_functions.LossFunctionConfig(
        loss_function_type=config_params.get("loss_function_type")
    )
