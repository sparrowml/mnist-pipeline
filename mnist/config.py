"""
Config with sensible defaults
"""
from typing import Tuple

from dataclasses import dataclass


@dataclass
class MnistConfig:
    # Configurable

    ## Dataset
    seed: int = 12345
    batch_size: int = 128
    n_dataset_threads: int = 4
    shuffle_buffer_size: int = 1024

    ## Model
    n_classes: int = 10
    dropout1_rate: float = 0.25
    dropout2_rate: float = 0.5

    ## Train
    n_epochs: int = 1
    verbose: bool = True

    # Fixed
    image_height: int = 28
    image_width: int = 28
    n_channels: int = 1
    n_train_samples: int = 60000
    n_test_samples: int = 10000

    # Helpers
    @classmethod
    def image_shape(cls) -> Tuple[int, int, int]:
        return (
            cls.image_height,
            cls.image_width,
            cls.n_channels,
        )

    @classmethod
    def steps_per_epoch(cls) -> int:
        return cls.n_train_samples // cls.batch_size

    @classmethod
    def validation_steps(cls) -> int:
        return cls.n_test_samples // cls.batch_size
