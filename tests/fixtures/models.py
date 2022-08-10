# -*- coding: utf-8 -*-
"""Models fixtures module.

Created on: 9/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import (
    Any,
    Dict,
)

from word2vect.ml import models


def get_model_artifacts(model_type: models.ModelType) -> Dict[str, Any]:
    """Get model artifacts based on the provided model type.

    Args:
        model_type: model type.

    Returns:
        model artifacts.
    """
    _implementations = {
        models.ModelType.WORD2VECT: {
            "model_name": "word2vect_vtest",
            "gradient_clipping_value": 1,
        }
    }

    model_artifacts = _implementations.get(model_type, None)

    return model_artifacts


def get_model_config(
    model_type: models.ModelType, model_artifacts: Dict[str, Any]
) -> models.ModelConfig:
    """Get model config using the provided model type and artifacts.

    Args:
        model_type: model type.
        model_artifacts: model artifacts.

    Returns:
        model_config: model configuration.
    """
    return models.ModelConfig(
        model_type=model_type,
        model_name=model_artifacts.get("model_name"),
        gradient_clipping_value=model_artifacts.get("gradient_clipping_value"),
    )
