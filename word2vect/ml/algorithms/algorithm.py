# -*- coding: utf-8 -*-
"""Algorithm interface.

Created on: 17/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from abc import (
    ABC,
    abstractmethod,
)

from word2vect.ml import tracker


class Algorithm(ABC):
    """Algorithm abstract class."""

    @abstractmethod
    def run(self, training_tracker: tracker.TrainingTracker) -> None:
        """Run training algorithm.

        Args:
            training_tracker: training tracker.

        Returns:
            None.
        """
        raise NotImplementedError()
