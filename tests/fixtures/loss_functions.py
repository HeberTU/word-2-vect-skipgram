# -*- coding: utf-8 -*-
"""Loss function fixtures.

Created on: 21/7/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import enum
from typing import Dict

import torch


class LossArtifactsType(enum.Enum):
    """Available loss artifacts types."""

    INTERFACE = enum.auto()
    NLLLOSS = enum.auto()


def get_loss_artifacts(
    loss_artifacts_type: LossArtifactsType,
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
        LossArtifactsType.INTERFACE: {
            "loss": torch.tensor([1]),
            "prediction": torch.tensor([4, 5, 6]),
            "target": torch.tensor([4, 5, 6]),
        },
        LossArtifactsType.NLLLOSS: {
            "loss": 1.8308,
            "prediction": nllloss_predictions,
            "target": torch.tensor([1, 0, 4]),
        },
    }

    loss_artifacts = _implementations.get(loss_artifacts_type, None)

    return loss_artifacts
