# -*- coding: utf-8 -*-
"""Word-2-vect model implementation.

Created on: 3/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import datetime
import os
import pathlib
from typing import (
    Optional,
    Type,
)

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from word2vect.ml import (
    loss_functions,
    metrics,
    tracker,
)

from .model_interface import (
    BatchData,
    ModelConfig,
    NNModel,
)


class Word2VectModel(NNModel):
    """Word 2 Vect Model."""

    def __init__(
        self,
        network: nn.Module,
        model_metrics: metrics.ModelMetrics,
        model_config: ModelConfig,
        optimizer: Optimizer,
        loss_function: loss_functions.LossFunction,
    ):
        """Instantiate a Word 2 vect model.

        Args:
            network: Neural network.
            model_metrics: Metric that will be logged.
            model_config: Model configuration data structure.
            optimizer: Model optimizer.
            loss_function: loss function.
        """
        self.network = network
        self.model_metrics = model_metrics
        self.model_name = model_config.model_name
        self.loss_function = loss_function
        self._optimizer = optimizer
        self._gradient_clipping_value = model_config.gradient_clipping_value
        self.model_config = model_config

        self._step: int = 0
        weights_path = model_config.network_weights_path
        path = pathlib.Path(__file__).parents[3] / "params"
        self.params_dir = weights_path if weights_path is not None else path

        self._initialize_repr_metrics()

        self.initial_repr = self.repr

        self.training_stats = loss_functions.TrainingStats()

    def _initialize_repr_metrics(self):
        """Create model representation."""
        self.model_metrics._models_repr = self.repr

    def forward(
        self,
        batch_data: BatchData,
        stage: tracker.Stage = tracker.Stage.SERVE,
    ) -> torch.Tensor:
        """Perform neural network pass using the given stage config.

        Args:
            batch_data: batch data.
            stage: An enum than can be TRAIN or SERVE. If SERVE the gradients
            computation is deactivated.

        Returns:
            predictions: A torch tensor containing network predictions.
        """
        if stage == tracker.Stage.SERVE:
            self.network.eval()
            with torch.no_grad():
                predictions = self.network.forward(batch_data)
        elif stage == tracker.Stage.SERVE:
            self.network.train()
            predictions = self.network.forward(batch_data)
        else:
            raise ValueError(f"{stage} is not a valid Stage.")

        return predictions

    def learn(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
    ) -> torch.Tensor:
        """Update the weights of the underlying neural network based on loss.

        Args:
            result: network results produced by a forward pass.
            ground_truth: Ground truth data.

        Returns:
            loss: model loss.
        """
        ground_truth = self.preprocess_ground_truth(
            ground_truth, stage=tracker.Stage.TRAIN
        )
        self._optimizer.zero_grad()
        loss = self.loss_function.compute(result, ground_truth)
        loss.backward()

        if self._gradient_clipping_value:
            nn.utils.clip_grad_norm_(
                parameters=self.network.parameters(),
                max_norm=self._gradient_clipping_value,
            )
        self._optimizer.step()

        return loss

    def evaluate(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
        loss: Optional[torch.Tensor] = None,
        stage: tracker.Stage = tracker.Stage.TRAIN,
    ) -> None:
        """Measure model performance using the model metrics.

        Args:
            result: network results produced by a forward pass.
            ground_truth: Ground truth data.
            loss: Loss values.
            stage: model stage.

        Returns:
            None.
        """
        ground_truth = self.preprocess_ground_truth(ground_truth, stage)
        result = self.preprocess_result(result, stage)

        self.model_metrics.update(result, ground_truth)
        if loss:
            self.training_stats.update(
                loss=loss,
                prediction=result.prediction,
                target=ground_truth.target,
            )

    def get_model_result(
        self,
        predictions: torch.Tensor,
        stage: tracker.Stage = tracker.Stage.TRAIN,
    ) -> loss_functions.Result:
        """Create a result instance base on net predictions.

        Args:
            predictions: predictions made by the network.
            stage: model stage.

        Returns:
            results: model results.
        """
        return loss_functions.Result(predictions.squeeze(1))

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get model optimizer."""
        return self._optimizer

    def log(
        self,
        training_tracker: tracker.TrainingTracker,
        stage: tracker.Stage = tracker.Stage.TRAIN,
    ) -> None:
        """Log metrics and training stats.

        Args:
            training_tracker: training tracker.
            stage: model stage

        Returns:
            None.
        """
        self.step += 1
        if self.training_stats.log_every_n_steps and (
            self.step % self.training_stats.log_every_n_steps == 0
        ):

            self._log_learning_stats(training_tracker)

        if self.model_metrics.log_every_n_steps and (
            self.step % self.model_metrics.log_every_n_steps == 0
        ):

            if stage == tracker.Stage.SERVE:
                self._log_model_metrics(training_tracker)

            if stage == tracker.Stage.TRAIN:
                training_tracker.log_learning_rate(
                    self.optimizer, self.model_name, self.step
                )
                training_tracker.log_weights(
                    self.network, self.model_name, self.step
                )
                training_tracker.log_grads(
                    self.network, self.model_name, self.step
                )

    def _log_learning_stats(
        self, training_tracker: Type[tracker.TrainingTracker]
    ) -> None:
        """Log the training stats.

        Args:
            training_tracker: Training tracker.

        Returns:
            None
        """
        training_tracker.log_training_stats(
            training_stats=self.training_stats,
            model_name=self.model_name,
            step=self.step,
        )
        self.training_stats.flush()

    def _log_model_metrics(
        self, training_tracker: Type[tracker.TrainingTracker]
    ) -> None:
        """Log the model metric.

        Args:
            training_tracker: Training tracker.

        Returns:
            None
        """
        training_tracker.log_metrics(
            model_metrics=self.model_metrics,
            model_name=self.model_name,
            step=self.step,
        )
        self.model_metrics.flush()

    def compute_loss(
        self,
        result: loss_functions.Result,
        ground_truth: loss_functions.GroundTruth,
    ) -> torch.Tensor:
        """Compute loss from model results.

        Args:
            result: network results produced by a forward pass.
            ground_truth:  Ground truth data.

        Returns:
            loss: model loss.
        """
        return self.loss_function.compute(result, ground_truth)

    def save_nn_model_parameters(self) -> None:
        """Save model torch params."""
        version = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        hashed_repr = self.initial_repr.hashed_repr

        params_filename = self.params_dir / hashed_repr / version + ".pt"
        obj_repr_filename = self.params_dir / hashed_repr / "object_repr.txt"

        os.makedirs(params_filename.parent, exist_ok=True)

        if not obj_repr_filename.is_file():
            self._write_repr_into_file(obj_repr_filename)

        torch.save(self.network.state_dict(), params_filename)

    def _write_repr_into_file(self, filename: pathlib.Path) -> None:
        """Write human-readable representation of the model.

        Args:
            filename: file location.

        Returns:
            None.
        """
        with open(filename, "w") as f:
            f.write(self.initial_repr.raw_repr)

    def load_nn_model_parameters(self) -> None:
        """Load model parameters."""
        model_base_path = self.params_dir / self.initial_repr.hashed_repr
        files = [file.name for file in model_base_path.rglob(pattern="*.pt")]
        if len(files) == 0:
            return None
        files.sort()
        params_file = files[-1]
        self.network.load_state_dict(
            state_dict=torch.load(model_base_path / params_file)
        )

    def preprocess_ground_truth(
        self, ground_truth: loss_functions.GroundTruth, stage: tracker.Stage
    ) -> loss_functions.GroundTruth:
        """Preprocess ground truth.

        Args:
            ground_truth: Ground truth data.
            stage: model stage.

        Returns:
            ground_truth: preprocess ground truth.
        """
        return ground_truth

    def preprocess_result(
        self, result: loss_functions.Result, stage: tracker.Stage
    ) -> loss_functions.Result:
        """Preprocess results.

        Args:
            result: network results produced by a forward pass.
            stage: model stage.

        Returns:
            results: preprocess results.
        """
        return result

    def __repr__(self):
        """Create a human-readable model representation."""
        args = [
            f"\nnetwork={repr(self.network)}",
            f"\nmodel_metrics={repr(self.model_metrics)}",
            f"\nmodel_config={repr(self.model_config)}",
            f"\noptimizer={repr(self._optimizer)}",
            f"\nloss_function={repr(self.loss_function)}",
        ]
        return f"DeterministicModel({', '.join(args)})"
