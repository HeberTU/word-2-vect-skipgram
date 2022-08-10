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
            [
                -2.9758,
                -3.4118,
                -3.2990,
                -3.1179,
                -2.8254,
                -2.2966,
                -3.9007,
                -3.4883,
                -3.3347,
                -2.6514,
                -3.8433,
                -3.5855,
                -2.5222,
                -2.8869,
                -2.9117,
                -3.2355,
                -2.7456,
                -2.5198,
                -3.3594,
                -2.8192,
            ],
            [
                -3.2224,
                -3.0172,
                -3.4387,
                -2.9950,
                -2.7580,
                -2.3514,
                -3.4383,
                -3.2970,
                -3.1167,
                -2.9918,
                -3.5133,
                -3.4017,
                -2.8122,
                -3.0139,
                -3.0974,
                -3.1655,
                -2.7351,
                -2.4954,
                -3.3967,
                -2.7132,
            ],
            [
                -3.1022,
                -3.5239,
                -3.5689,
                -3.4325,
                -3.0668,
                -2.0905,
                -3.8764,
                -3.2532,
                -3.7994,
                -2.2612,
                -3.9485,
                -3.8657,
                -2.7106,
                -3.2091,
                -2.4925,
                -3.5278,
                -2.7234,
                -2.1740,
                -3.8109,
                -3.0326,
            ],
            [
                -3.0902,
                -2.8508,
                -3.0722,
                -2.9528,
                -3.2653,
                -2.6618,
                -3.3846,
                -3.2274,
                -3.0323,
                -3.1094,
                -3.2217,
                -3.1213,
                -2.9287,
                -2.6889,
                -3.1889,
                -2.9907,
                -2.7480,
                -2.6924,
                -3.4066,
                -2.7806,
            ],
            [
                -2.9063,
                -2.5868,
                -3.0509,
                -3.1359,
                -3.3152,
                -2.4150,
                -3.7158,
                -2.9882,
                -2.8487,
                -3.3810,
                -3.5249,
                -3.1741,
                -2.8412,
                -2.8812,
                -3.0190,
                -3.3230,
                -2.7323,
                -2.7399,
                -3.4230,
                -2.9172,
            ],
            [
                -3.1627,
                -3.0081,
                -3.3343,
                -3.1402,
                -3.2512,
                -2.3593,
                -3.4488,
                -3.0955,
                -3.4232,
                -2.6723,
                -3.4286,
                -3.4526,
                -2.9388,
                -2.9104,
                -2.7815,
                -3.1889,
                -2.6557,
                -2.3675,
                -3.7266,
                -2.9045,
            ],
            [
                -2.9153,
                -3.1238,
                -3.0633,
                -3.2056,
                -3.2857,
                -2.4976,
                -3.5646,
                -3.1739,
                -3.2157,
                -2.7856,
                -3.4099,
                -3.2050,
                -2.7725,
                -2.8856,
                -2.9004,
                -3.2422,
                -2.8073,
                -2.4691,
                -3.3907,
                -2.8446,
            ],
            [
                -2.6554,
                -3.4540,
                -2.9105,
                -3.4206,
                -3.1693,
                -2.5457,
                -3.7812,
                -3.3125,
                -3.0804,
                -2.8192,
                -3.5648,
                -3.0493,
                -2.5258,
                -3.0980,
                -2.9709,
                -3.4909,
                -3.0569,
                -2.4709,
                -2.9940,
                -2.7884,
            ],
            [
                -2.9338,
                -3.2138,
                -3.1682,
                -3.2511,
                -3.1034,
                -2.4091,
                -3.5806,
                -3.2048,
                -3.2178,
                -2.7612,
                -3.5033,
                -3.2595,
                -2.7169,
                -3.0372,
                -2.8877,
                -3.3359,
                -2.8442,
                -2.3948,
                -3.3307,
                -2.8078,
            ],
            [
                -2.9659,
                -2.6754,
                -3.0529,
                -3.0681,
                -3.2965,
                -2.4980,
                -3.5959,
                -3.0673,
                -2.9103,
                -3.2800,
                -3.4139,
                -3.1514,
                -2.8681,
                -2.8083,
                -3.0731,
                -3.2020,
                -2.7333,
                -2.7189,
                -3.4144,
                -2.8661,
            ],
        ]
    )

    _implementations = {
        loss_functions.LossFunctionType.INTERFACE: {
            "loss": torch.tensor([1]),
            "log_prob": torch.tensor(
                [0, 1, 2, 0, 1, 2]  # Not important, it is a placeholder.
            ),
            "prediction": torch.tensor([0, 1, 2, 0, 1, 2]),
            "target": torch.tensor([0, 2, 1, 0, 0, 1]),
        },
        loss_functions.LossFunctionType.NLLLOSS: {
            "loss": 2.814249,
            "log_prob": nllloss_predictions,
            "prediction": torch.tensor([5, 5, 5, 5, 5, 5, 17, 17, 17, 5]),
            "target": torch.tensor([3, 5, 1, 5, 2, 5, 12, 17, 15, 5]),
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
