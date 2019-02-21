"""
Config with sensible defaults
"""
from typing import Tuple

import yaml
from dataclasses import dataclass


@dataclass
class MnistConfig:
    # Model
    version: str = '0.0.1'
    n_classes: int = 10
    dropout1_rate: float = 0.25
    dropout2_rate: float = 0.5

    # Train
    seed: int = 12345
    batch_size: int = 128
    n_epochs: int = 1
    verbose: bool = True

    # Nuisance parameters
    n_dataset_threads: int = 4
    shuffle_buffer_size: int = 1024

    # Fixed
    image_height: int = 28
    image_width: int = 28
    n_channels: int = 1
    n_train_samples: int = 60000
    n_test_samples: int = 10000

    # Derived values
    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (
            self.image_height,
            self.image_width,
            self.n_channels,
        )

    @property
    def steps_per_epoch(self) -> int:
        return self.n_train_samples // self.batch_size

    @property
    def validation_steps(self) -> int:
        return self.n_test_samples // self.batch_size

    # Initialization
    @classmethod
    def from_yaml(cls: 'MnistConfig', path: str) -> 'MnistConfig':
        with open(path) as configfile:
            configdict = yaml.load(configfile)
        return MnistConfig(**configdict)
